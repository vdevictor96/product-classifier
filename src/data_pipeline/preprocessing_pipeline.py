"""Module for preprocessing the product data for BERT-based NLP tasks."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from src.data_pipeline import utils

logger = utils.get_logger(__name__)


class PreprocessingPipeline:
    """
    A reusable class to preprocess product data for BERT-based NLP tasks.

    Methods:
        preprocess(df: DataFrame) -> DataFrame:
            The entire pipeline including cleaning and text processing.
    """

    def __init__(self):
        """
        Initializes the preprocessing pipeline with a BERT tokenizer.

        """
        self.ordered_columns_to_keep = [
            "title", "description", "feature", "brand", "main_cat"]
        self.combined_text_column = "combined_text"

    def _clean_data(self, df: DataFrame, ordered_columns_to_keep=None) -> DataFrame:
        """
        Clean the input DataFrame by removing unnecessary columns and handling missing values.

        Args:
            df (DataFrame): Input DataFrame.
            ordered_columns_to_keep (list, optional): List of columns to keep in the DataFrame in the desired order. 
                                                  If None, keep all columns in their original order.
        Returns:
            DataFrame: Cleaned DataFrame.
        """
        try:
            # Ensure "ordered_columns_to_keep" is a subset of the actual DataFrame columns
            if ordered_columns_to_keep is not None:
                actual_columns = df.columns
                ordered_columns_to_keep = [
                    col for col in ordered_columns_to_keep if col in actual_columns]
                df = df.select(*ordered_columns_to_keep)

            return df
        except Exception as e:
            logger.error("Failed to clean data: %s", e, exc_info=True)
            raise

    def _process_text(self, df: DataFrame, combined_text_column="combined_text") -> DataFrame:
        """
        Process the text with BERT tokenization.

        Args:
            df (DataFrame): Input dataframe with product data.

        Returns:
            DataFrame: DataFrame with tokenized text column.
        """
        try:
            df = self._combine_text_fields(df, combined_text_column)
            # Drop the original columns
            columns_to_keep = [combined_text_column, "main_cat"]
            return df.select(*columns_to_keep)
        except Exception as e:
            logger.error("Failed to process text: %s", e, exc_info=True)
            raise

    def _combine_text_fields(self, df: DataFrame, combined_text_column="combined_text") -> DataFrame:
        """
        Combines text fields into a single field for model input.

        Args:
            df (DataFrame): Input DataFrame.
            combined_text_column (str, optional): Name of the new column with the combined text. Defaults to "combined_text".

        Returns:
            DataFrame: DataFrame with combined text field.
        """

        # Define a UDF to combine all columns into a single string
        combine_udf = udf(
            lambda *cols: " ".join([str(col) for col in cols if col is not None]), StringType())

        # Get all column names from the DataFrame except the label column "main_cat"
        column_names = [c for c in df.columns if c != "main_cat"]

        # Apply the UDF to combine all columns
        return df.withColumn(combined_text_column, combine_udf(*[col(c) for c in column_names]))

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Preprocess the entire pipeline including cleaning and text processing.

        Args:
            df (DataFrame): Input dataframe with product data.

        Returns:
            DataFrame: Preprocessed dataframe ready for training or inference.
        """
        try:
            logger.info("Starting preprocessing pipeline.")
            logger.info("Cleaning the data.")
            cleaned_df = self._clean_data(df, self.ordered_columns_to_keep)
            logger.info("Processing the text data.")
            preprocessed_df = self._process_text(
                cleaned_df, self.combined_text_column)
            logger.info("Preprocessing pipeline completed.")
            return preprocessed_df
        except Exception as e:
            logger.error("Preprocessing pipeline failed: %s", e, exc_info=True)
            raise
