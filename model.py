"""Feature engineering, training, evaluation and inference for salary prediction.

Pipeline:  SQLite -> regex skill flags -> (one-hot cats + skill flags) -> XGBoost

Two kinds of output are produced:

* a **point estimate** (squared-error XGBoost), and
* a **prediction interval** from *conformalized quantile regression* (CQR):
  two quantile models (10th / 90th percentile) are widened by a margin computed
  on a held-out calibration set, which gives finite-sample coverage guarantees
  (~80 % of true salaries fall inside the interval) without distributional assumptions.

The entire sklearn Pipeline objects are serialised with joblib, so inference can
never drift from training-time preprocessing. Optional MLflow tracking: ``--track``.

Usage:
    python model.py                       # train on simulated data
    python model.py --source remotive     # train on real Remotive postings
    python model.py --track               # also log the run to MLflow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:  # graceful degradation if xgboost is unavailable
    XGBRegressor = None  # type: ignore[assignment,misc]

from config import (
    BASE_DIR, DB_PATH, EXPERIENCE_LEVELS, METRICS_PATH, MODEL_PATH, SKILL_PATTERNS,
    extract_skill_flags, setup_logging, skill_column,
)
from database import fetch_jobs_df

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES: list[str] = ["role_category", "location", "experience_level"]
SKILL_FEATURES: list[str] = [skill_column(s) for s in SKILL_PATTERNS]
FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + SKILL_FEATURES
TARGET = "salary_mid"
TRACKED_ROLES = ["Data Scientist", "Data Engineer", "ML Engineer", "Software Engineer"]
DEFAULT_MIN_ROWS = 100
INTERVAL_ALPHA = 0.2  # 1 - alpha = 80 % target coverage


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_skill_flags(df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
    """Append one binary ``skill_*`` column per tracked skill (regex extraction)."""
    base = df.drop(columns=[c for c in SKILL_FEATURES if c in df.columns])
    rows = [extract_skill_flags(text) for text in df[text_column].fillna("")]
    flags = pd.DataFrame(rows, index=df.index).rename(columns=skill_column)
    flags = flags.reindex(columns=SKILL_FEATURES, fill_value=0).astype(int)
    return pd.concat([base, flags], axis=1)


def load_training_frame(db_path: Path = DB_PATH, source: str = "simulated") -> pd.DataFrame:
    """Load labelled rows (disclosed salary, tracked role) with skill flags attached.

    Args:
        source: ``"simulated"``, ``"remotive"`` or ``"all"``. Mixing real and
            synthetic rows contaminates the target, so the default is one source.
    """
    df = fetch_jobs_df(db_path)
    total = len(df)
    if source != "all":
        df = df[df["source"] == source]
    df = df.dropna(subset=[TARGET])
    df = df[df["role_category"].isin(TRACKED_ROLES)]
    logger.info("Loaded %d jobs; %d usable for training (source=%s, salary disclosed)", total, len(df), source)
    return add_skill_flags(df)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals (unseen-safe) and pass skill flags through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("skills", "passthrough", SKILL_FEATURES),
        ]
    )


def build_pipeline(model_type: str = "xgboost", random_state: int = 42) -> tuple[Pipeline, str]:
    """Build the point-estimate pipeline.

    Returns:
        ``(pipeline, resolved_model_type)``; falls back to RandomForest if
        XGBoost was requested but is not installed.
    """
    if model_type == "xgboost" and XGBRegressor is not None:
        regressor: Any = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, tree_method="hist",
            random_state=random_state, n_jobs=-1,
        )
        resolved = "xgboost"
    else:
        if model_type == "xgboost":
            logger.warning("xgboost not installed - falling back to RandomForest")
        regressor = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
        resolved = "random_forest"
    return Pipeline([("preprocess", build_preprocessor()), ("model", regressor)]), resolved


def _quantile_pipeline(alpha: float, random_state: int) -> Pipeline:
    """Pipeline that predicts the ``alpha``-quantile of salary."""
    if XGBRegressor is not None:
        regressor: Any = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha, n_estimators=300,
            learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85,
            tree_method="hist", random_state=random_state, n_jobs=-1,
        )
    else:
        regressor = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, n_estimators=300, learning_rate=0.05,
            max_depth=3, random_state=random_state,
        )
    return Pipeline([("preprocess", build_preprocessor()), ("model", regressor)])


# --------------------------------------------------------------------------- #
# Conformalized quantile regression (CQR)
# --------------------------------------------------------------------------- #
def _raw_band(models: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Un-calibrated quantile band with crossing quantiles repaired."""
    lo, hi = models["lower"].predict(X), models["upper"].predict(X)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def fit_interval_models(X: pd.DataFrame, y: pd.Series, alpha: float = INTERVAL_ALPHA,
                        calibration_fraction: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    """Fit lower/upper quantile models and a conformal margin.

    Steps: split into fit/calibration -> train quantile models on *fit* ->
    compute conformity scores ``max(lo - y, y - hi)`` on *calibration* ->
    take the ``ceil((n+1)(1-alpha))/n`` empirical quantile as the margin ``q``.
    Final interval: ``[lo - q, hi + q]``.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=calibration_fraction, random_state=random_state)
    models: dict[str, Any] = {
        "lower": _quantile_pipeline(alpha / 2, random_state).fit(X_fit, y_fit),
        "upper": _quantile_pipeline(1 - alpha / 2, random_state).fit(X_fit, y_fit),
    }
    lo, hi = _raw_band(models, X_cal)
    scores = np.maximum(lo - y_cal.to_numpy(), y_cal.to_numpy() - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    models["conformal_q"] = float(np.quantile(scores, level, method="higher"))
    models["alpha"] = alpha
    models["n_calibration"] = int(n)
    return models


def interval_bounds(models: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated ``(lower, upper)`` salary bounds for each row of ``X``."""
    lo, hi = _raw_band(models, X)
    return np.maximum(lo - models["conformal_q"], 0.0), hi + models["conformal_q"]


# --------------------------------------------------------------------------- #
# Evaluation / tracking helpers
# --------------------------------------------------------------------------- #
def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _feature_importance(pipeline: Pipeline, top_n: int = 20) -> list[dict[str, float]]:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"feature": n.split("__", 1)[-1], "importance": float(v)} for n, v in ranked]


def _log_to_mlflow(params: dict[str, Any], metrics: dict[str, float], artifacts: list[Path]) -> None:
    """Log one run to MLflow (optional dependency; local SQLite store by default)."""
    try:
        import mlflow
    except ImportError:
        logger.warning("--track requested but mlflow is not installed (pip install mlflow); skipping")
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}"))
    mlflow.set_experiment("tech-salary-model")
    with mlflow.start_run(run_name=f"{params.get('model_type')}-{params.get('trained_on')}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        logger.info("Logged run to MLflow (%s)", mlflow.get_tracking_uri())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_save(db_path: Path = DB_PATH, model_path: Path = MODEL_PATH,
                   metrics_path: Path = METRICS_PATH, model_type: str = "xgboost",
                   test_size: float = 0.2, random_state: int = 42, source: str = "simulated",
                   min_rows: int = DEFAULT_MIN_ROWS, track: bool = False) -> dict[str, Any]:
    """Train, evaluate on a hold-out set, and persist the model bundle.

    Returns:
        The metrics dictionary (also written to ``metrics_path``).

    Raises:
        ValueError: If there are too few labelled rows to train responsibly.
    """
    df = load_training_frame(db_path, source)
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} labelled rows for source '{source}' (need >= {min_rows}). "
            "Scrape more data first (real boards disclose salary rarely)."
        )

    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline, resolved = build_pipeline(model_type, random_state)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=KFold(5, shuffle=True, random_state=random_state),
        scoring="neg_root_mean_squared_error",
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    # Honest interval evaluation: calibrate on train only, measure coverage on the untouched test set.
    eval_models = fit_interval_models(X_train, y_train, INTERVAL_ALPHA, random_state=random_state)
    lo, hi = interval_bounds(eval_models, X_test)
    coverage = float(np.mean((y_test.to_numpy() >= lo) & (y_test.to_numpy() <= hi)))
    naive_coverage = float(np.mean(np.abs(y_test.to_numpy() - preds) <= mae))

    metrics: dict[str, Any] = {
        "model_type": resolved,
        "trained_on": source,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": _rmse(y_test, preds),
        "mae": mae,
        "r2": float(r2_score(y_test, preds)),
        "cv_rmse_mean": float(-cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "baseline_rmse": _rmse(y_test, np.full(len(y_test), y_train.mean())),
        "interval_target_coverage": 1 - INTERVAL_ALPHA,
        "interval_coverage": coverage,
        "interval_mean_width": float(np.mean(hi - lo)),
        "naive_mae_band_coverage": naive_coverage,
    }
    logger.info("Model: %s | source=%s | train=%d test=%d", resolved, source, metrics["n_train"], metrics["n_test"])
    logger.info("Hold-out RMSE: $%.0f | MAE: $%.0f | R2: %.3f", metrics["rmse"], mae, metrics["r2"])
    logger.info("5-fold CV RMSE: $%.0f (+/- $%.0f) | mean-baseline RMSE: $%.0f",
                metrics["cv_rmse_mean"], metrics["cv_rmse_std"], metrics["baseline_rmse"])
    logger.info("80%% interval: CQR coverage %.1f%% (width $%.0f) vs naive +/-MAE band coverage %.1f%%",
                coverage * 100, metrics["interval_mean_width"], naive_coverage * 100)

    # Production artifacts: refit on ALL labelled data.
    pipeline.fit(X, y)
    interval_models = fit_interval_models(X, y, INTERVAL_ALPHA, random_state=random_state)
    importance = _feature_importance(pipeline)
    for item in importance[:8]:
        logger.info("  top feature %-28s %.3f", item["feature"], item["importance"])

    bundle = {
        "pipeline": pipeline,
        "interval_models": interval_models,
        "feature_columns": FEATURE_COLUMNS,
        "skills": list(SKILL_PATTERNS),
        "categories": {
            "role_category": sorted(df["role_category"].unique()),
            "location": sorted(df["location"].unique()),
            "experience_level": [lvl for lvl in EXPERIENCE_LEVELS if lvl in set(df["experience_level"])],
        },
        "role_mean_salary": df.groupby("role_category")[TARGET].mean().round(0).to_dict(),
        "metrics": metrics,
        "feature_importance": importance,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    metrics_path.write_text(json.dumps({**metrics, "feature_importance": importance}, indent=2))
    logger.info("Saved model bundle -> %s", model_path)

    if track:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        params = {"model_type": resolved, "trained_on": source, "test_size": test_size,
                  "random_state": random_state, "n_features": len(FEATURE_COLUMNS)}
        _log_to_mlflow(params, numeric, [model_path, metrics_path])
    return metrics


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the serialised model bundle.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model at {model_path}. Run `python model.py` first.")
    return joblib.load(model_path)


def _feature_row(role: str, location: str, experience: str, skills: Sequence[str]) -> dict[str, Any]:
    chosen = set(skills)
    unknown = chosen - set(SKILL_PATTERNS)
    if unknown:
        logger.warning("Ignoring unknown skills: %s", sorted(unknown))
    row: dict[str, Any] = {"role_category": role, "location": location, "experience_level": experience}
    row.update({skill_column(s): int(s in chosen) for s in SKILL_PATTERNS})
    return row


def predict_salary(bundle: dict[str, Any], role: str, location: str, experience: str,
                   skills: Sequence[str]) -> float:
    """Predict the annual market-rate salary (USD) for a profile."""
    frame = pd.DataFrame([_feature_row(role, location, experience, skills)])[bundle["feature_columns"]]
    return float(max(bundle["pipeline"].predict(frame)[0], 0.0))


def predict_with_interval(bundle: dict[str, Any], role: str, location: str, experience: str,
                          skills: Sequence[str]) -> tuple[float, float, float]:
    """Return ``(point, lower, upper)``; the interval is a conformal 80 % band.

    Falls back to ``point +/- MAE`` for bundles saved before intervals existed.
    """
    frame = pd.DataFrame([_feature_row(role, location, experience, skills)])[bundle["feature_columns"]]
    point = float(max(bundle["pipeline"].predict(frame)[0], 0.0))
    if "interval_models" in bundle:
        lo, hi = interval_bounds(bundle["interval_models"], frame)
        lower, upper = float(lo[0]), float(hi[0])
    else:
        mae = bundle["metrics"]["mae"]
        lower, upper = max(point - mae, 0.0), point + mae
    return point, min(lower, point), max(upper, point)


def skill_uplift(bundle: dict[str, Any], role: str, location: str, experience: str,
                 skills: Sequence[str], top_n: int = 5) -> pd.DataFrame:
    """Estimate the salary change from adding each skill the profile lacks."""
    base = predict_salary(bundle, role, location, experience, skills)
    candidates = [s for s in bundle["skills"] if s not in set(skills)]
    if not candidates:
        return pd.DataFrame(columns=["skill", "uplift"])
    rows = [_feature_row(role, location, experience, [*skills, c]) for c in candidates]
    preds = bundle["pipeline"].predict(pd.DataFrame(rows)[bundle["feature_columns"]])
    out = pd.DataFrame({"skill": candidates, "uplift": preds - base})
    return out.sort_values("uplift", ascending=False).head(top_n).reset_index(drop=True)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Train the salary prediction model")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--model-type", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--source", choices=["simulated", "remotive", "all"], default="simulated")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--track", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()
    train_and_save(Path(args.db), model_type=args.model_type, test_size=args.test_size,
                   source=args.source, min_rows=args.min_rows, track=args.track)


if __name__ == "__main__":
    main()
