import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


def preprocess(df, zero_cols, q1, q3, n_neighbors):
    df = df.copy()
    for col in zero_cols:
        if col in df.columns:
            df[col] = np.where(df[col] == 0, np.nan, df[col])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Outcome" in num_cols:
        num_cols.remove("Outcome")
    scaler_knn = RobustScaler()
    scaled = scaler_knn.fit_transform(df[num_cols])
    imputed = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    df[num_cols] = scaler_knn.inverse_transform(imputed)
    for col in num_cols:
        low = df[col].quantile(q1)
        high = df[col].quantile(q3)
        iqr = high - low
        df[col] = df[col].clip(lower=low - 1.5 * iqr, upper=high + 1.5 * iqr)
    return df


def build_features(df):
    df = df.copy()
    df["NEW_AGE_CAT"] = "young"
    df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
    df.loc[df["Age"] >= 50, "NEW_AGE_CAT"] = "senior"
    df["NEW_BMI"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                           labels=["Underweight", "Healthy", "Overweight", "Obese"]).astype(str)
    df["NEW_GLUCOSE"] = pd.cut(df["Glucose"], bins=[0, 140, 200, 300],
                               labels=["Normal", "Prediabetes", "Diabetes"]).astype(str)
    c1 = [(df["BMI"] < 18.5) & (df["Age"] < 50),(df["BMI"] < 18.5) & (df["Age"] >= 50),
          ((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] < 50),
          ((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50),
          ((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] < 50),
          ((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50),
          (df["BMI"] >= 30) & (df["Age"] < 50),(df["BMI"] >= 30) & (df["Age"] >= 50)]
    v1 = ["underweightmature","underweightsenior","healthymature","healthysenior",
          "overweightmature","overweightsenior","obesemature","obesesenior"]
    df["NEW_AGE_BMI_NOM"] = np.select(c1, v1, default="other")
    c2 = [(df["Glucose"] < 70) & (df["Age"] < 50),(df["Glucose"] < 70) & (df["Age"] >= 50),
          ((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] < 50),
          ((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50),
          ((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] < 50),
          ((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50),
          (df["Glucose"] > 125) & (df["Age"] < 50),(df["Glucose"] > 125) & (df["Age"] >= 50)]
    v2 = ["lowmature","lowsenior","normalmature","normalsenior",
          "hiddenmature","hiddensenior","highmature","highsenior"]
    df["NEW_AGE_GLUCOSE_NOM"] = np.select(c2, v2, default="other")
    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")
    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]
    df.columns = [col.upper() for col in df.columns]
    return df


def encode_and_split(df, test_size, random_state):
    df = df.copy()
    target = "OUTCOME"

    # Converter tudo que nao e numero para string
    for col in df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).fillna("unknown")

    # Label encoding nas binarias
    binary_cols = [col for col in df.columns
                   if df[col].dtype == "O" and df[col].nunique() == 2 and col != target]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot nas demais categoricas
    cat_cols = [col for col in df.columns
                if df[col].dtype == "O" and col not in binary_cols and col != target]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Booleanas para int
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # GARANTIR que todas as colunas sao numericas antes de salvar
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Split
    X = df.drop(columns=[target])
    y = df[[target]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Scaling
    num_cols = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=num_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=num_cols)

    feature_columns = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, scaler, feature_columns
