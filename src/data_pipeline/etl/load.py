"""Module that loads the proprocessed data back to a safe storage. Last step in ETL pipeline."""
from pyspark.sql import DataFrame
from src.data_pipeline import utils

logger = utils.get_logger(__name__)


def to_parquet_file(df: DataFrame, output_path: str):
    """
    Loads the transformed data to the target location in Parquet format.

    Args:
        df (DataFrame): Transformed data.
        output_path (str): Path where the Parquet file will be saved.
    """
    try:
        df.write.mode('overwrite').parquet(output_path)
    except Exception as e:
        logger.error("Failed to load data to Parquet file at %s: %s",
                     output_path, e, exc_info=True)
        raise
