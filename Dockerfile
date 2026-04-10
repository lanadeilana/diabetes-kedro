FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY src/ ./src/
COPY api/ ./api/
COPY conf/ ./conf/
COPY data/06_models/ ./data/06_models/

RUN uv pip install --system kedro "kedro-datasets[pandas,pickle]" pandas numpy \
    scikit-learn xgboost lightgbm joblib fastapi "uvicorn[standard]" pyarrow

RUN uv pip install --system -e src/ --no-deps

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
