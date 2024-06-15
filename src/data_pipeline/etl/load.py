from pyspark.sql import DataFrame


def to_parquet_file(df: DataFrame, output_path: str):
    """
    Loads the transformed data to the target location in Parquet format.

    Args:
        df (DataFrame): Transformed data.
        output_path (str): Path where the Parquet file will be saved.
    """
    df.write.mode('overwrite').parquet(output_path)
