"""Module that takes care of the cleaning of the raw data in an ETL pipeline."""
from pyspark.sql import DataFrame


def clean_data(df: DataFrame, ordered_columns_to_keep=None) -> DataFrame:
    """
    Clean the input DataFrame by removing unnecessary columns and handling missing values.

    Args:
        df (DataFrame): Input DataFrame.
        ordered_columns_to_keep (list, optional): List of columns to keep in the DataFrame in the desired order. 
                                                  If None, keep all columns in their original order.
    Returns:
        DataFrame: Cleaned DataFrame.
    """
    # Ensure "ordered_columns_to_keep" is a subset of the actual DataFrame columns
    if ordered_columns_to_keep is not None:
        actual_columns = df.columns
        ordered_columns_to_keep = [col for col in ordered_columns_to_keep if col in actual_columns]
        df = df.select(*ordered_columns_to_keep)

    return df
