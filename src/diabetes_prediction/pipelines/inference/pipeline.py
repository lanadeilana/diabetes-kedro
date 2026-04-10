from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_inference, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_inference,
            inputs=[
                "diabetes_inference",
                "scaler",
                "feature_columns",
                "params:zero_as_nan_cols",
            ],
            outputs="inference_processed",
            name="preprocess_inference_node",
        ),
        node(
            func=predict,
            inputs=["inference_processed", "best_model", "params:prediction_threshold"],
            outputs="predictions",
            name="predict_node",
        ),
    ])
