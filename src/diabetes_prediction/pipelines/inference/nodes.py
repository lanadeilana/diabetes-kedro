import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def preprocess_inference(df, scaler, feature_columns, zero_cols):
    from diabetes_prediction.pipelines.data_engineering.nodes import build_features
    df = df.copy()
    for col in zero_cols:
        if col in df.columns:
            df[col] = np.where(df[col] == 0, np.nan, df[col])
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df = build_features(df)
    for col in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).fillna('unknown')
    binary_cols = [c for c in df.columns if df[c].dtype == 'O' and df[c].nunique() == 2]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    cat_cols = [c for c in df.columns if df[c].dtype == 'O' and c not in binary_cols]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in list(df.select_dtypes(include='bool').columns):
        df[col] = df[col].astype(int)
    for col in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_columns].astype(float)
    return df


def predict(df, model, threshold):
    arr = df.values.astype(float)
    proba = model.predict_proba(arr)[:, 1]
    labels = (proba >= threshold).astype(int)
    result = pd.DataFrame({'predicted_proba': proba.round(4), 'predicted_label': labels})
    logger.info('Inferencia: %d registros | %d positivos', len(labels), int(labels.sum()))
    return result