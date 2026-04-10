import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


def train_models(X_train, y_train, X_test, y_test, selected_model, lgbm_params, rf_params):
    """
    Treina os mesmos modelos do notebook do professor e retorna o melhor.
    """
    y_train_arr = y_train.values.ravel()
    y_test_arr = y_test.values.ravel()

    models = {
        "random_forest": RandomForestClassifier(**rf_params),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "svc": SVC(),
        "decision_tree": DecisionTreeClassifier(),
        "adaboost": AdaBoostClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(eval_metric="logloss", verbosity=0),
        "lightgbm": LGBMClassifier(**lgbm_params),
    }

    results = []
    trained = {}

    for name, model in models.items():
        logger.info("Treinando: %s", name)
        model.fit(X_train, y_train_arr)
        trained[name] = model

        y_pred = model.predict(X_test)
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test_arr, y_pred), 4),
            "recall": round(recall_score(y_test_arr, y_pred), 4),
            "precision": round(precision_score(y_test_arr, y_pred), 4),
            "f1": round(f1_score(y_test_arr, y_pred), 4),
            "auc": round(roc_auc_score(y_test_arr, y_pred), 4),
        })

    metrics_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    logger.info("\n%s", metrics_df.to_string(index=False))

    if selected_model not in trained:
        raise ValueError(f"Modelo '{selected_model}' invalido. Opcoes: {list(trained.keys())}")

    logger.info("Modelo selecionado: %s", selected_model)
    return trained[selected_model], metrics_df
