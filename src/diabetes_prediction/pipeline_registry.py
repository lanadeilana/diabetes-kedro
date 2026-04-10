from kedro.pipeline import Pipeline
from diabetes_prediction.pipelines import data_engineering, training, inference


def register_pipelines() -> dict[str, Pipeline]:
    de = data_engineering.create_pipeline()
    tr = training.create_pipeline()
    inf = inference.create_pipeline()

    return {
        "data_engineering": de,
        "training": tr,
        "inference": inf,
        "__default__": de + tr,
    }
