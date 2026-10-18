.PHONY: install data model run test schedule real track mlflow-ui clean

install:   ## install dependencies
	pip install -r requirements.txt pytest

data:      ## scrape a reproducible simulated snapshot
	python scraper.py --pages 40 --seed 42

real:      ## fetch REAL listings from Remotive (max ~4x/day; cached 6h)
	python scraper.py --source remotive

model:     ## train + evaluate + save
	python model.py

track:     ## train and log to MLflow (pip install mlflow)
	python model.py --track

mlflow-ui: ## browse experiments at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

run:       ## launch dashboard
	streamlit run app.py

schedule:  ## run scheduler (simulator only)
	python scheduler.py --no-remotive

test:
	python -m pytest -q tests

clean:
	rm -rf data models logs mlruns mlflow.db .pytest_cache
