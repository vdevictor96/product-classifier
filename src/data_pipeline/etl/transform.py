"""Module that takes care of preprocessing the raw data in an ETL pipeline."""

from pyspark.sql import DataFrame

from src.data_pipeline.preprocessing_pipeline import PreprocessingPipeline


def preprocess(df: DataFrame) -> DataFrame:
    """
    Preprocesses the raw data for BERT-based NLP tasks.

    Args:
        df (DataFrame): Input DataFrame with raw data.

    Returns:
        DataFrame: DataFrame with preprocessed data.
    """
    preprocessing_pipeline = PreprocessingPipeline()
    return preprocessing_pipeline.preprocess(df)
