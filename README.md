# Previsão de Diabetes com Kedro

Projeto desenvolvido para a disciplina de Data Science e Decisão (PADS - Insper).

Migração do notebook `diabetes-prediction.ipynb` para pipelines Kedro.

## Setup

```bash
pip install uv
uv venv --python 3.11
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Mac/Linux
uv pip install -e .
```

## Executar

```bash
# Pipeline completo (engenharia de dados + treinamento)
kedro run

# Visualizar o grafo
kedro viz

# Inferência
kedro run --pipeline inference

# API (extra)
uvicorn api.main:app --reload
```

## Pipelines

- **data_engineering**: limpeza, imputação KNN, feature engineering, encoding, split
- **training**: treina 9 modelos, salva o melhor (LightGBM por padrão)
- **inference**: aplica o modelo nos dados de inferência do professor
