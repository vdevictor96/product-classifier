"""Module that takes care of preprocessing the raw data in an ETL pipeline."""

from pyspark.sql import DataFrame

from src.data_pipeline.etl.preprocessing_pipeline import PreprocessingPipeline
from src.data_pipeline import utils

logger = utils.get_logger(__name__)


def preprocess(df: DataFrame) -> DataFrame:
    """
    Preprocesses the raw data for BERT-based NLP tasks.

    Args:
        df (DataFrame): Input DataFrame with raw data.

    Returns:
        DataFrame: DataFrame with preprocessed data.
    """
    try:
        preprocessing_pipeline = PreprocessingPipeline()
        return preprocessing_pipeline.preprocess(df)
    except Exception as e:
        logger.error("Failed to preprocess data: %s", e, exc_info=True)
        raise
