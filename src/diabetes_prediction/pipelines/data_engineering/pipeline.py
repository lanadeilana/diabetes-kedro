from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess, build_features, encode_and_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess,
            inputs=[
                "diabetes_modelling",
                "params:zero_as_nan_cols",
                "params:outlier_q1",
                "params:outlier_q3",
                "params:knn_imputer_neighbors",
            ],
            outputs="diabetes_cleaned",
            name="preprocess_node",
        ),
        node(
            func=build_features,
            inputs="diabetes_cleaned",
            outputs="diabetes_featured",
            name="build_features_node",
        ),
        node(
            func=encode_and_split,
            inputs=[
                "diabetes_featured",
                "params:test_size",
                "params:random_state",
            ],
            outputs=["X_train", "X_test", "y_train", "y_test", "scaler", "feature_columns"],
            name="encode_and_split_node",
        ),
    ])
