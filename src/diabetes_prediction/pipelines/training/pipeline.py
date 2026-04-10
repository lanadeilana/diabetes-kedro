from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_models,
            inputs=[
                "X_train", "y_train", "X_test", "y_test",
                "params:selected_model",
                "params:lgbm_params",
                "params:rf_params",
            ],
            outputs=["best_model", "model_metrics"],
            name="train_models_node",
        ),
    ])
