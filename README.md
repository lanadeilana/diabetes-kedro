# Diabetes Prediction with Kedro

**By Lana**

An end-to-end machine learning pipeline for diabetes prediction, built with Kedro for pipeline orchestration, FastAPI for model serving, and Docker for containerization.

## Setup

```bash
pip install uv
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
```

## Run

```bash
# Run the full pipeline
kedro run

# Visualize the pipeline
kedro viz

# Run inference pipeline only
kedro run --pipeline inference

# Start the API
uvicorn api.main:app --reload
```

## Pipelines

- **data_engineering** — data ingestion, cleaning, and feature engineering
- **training** — model training, evaluation, and artifact export
- **inference** — loads saved model and generates predictions via FastAPI
