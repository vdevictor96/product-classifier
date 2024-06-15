from pyspark.sql import DataFrame
from transformers import BertTokenizer
from src.data_pipeline import clean, transform, utils

logger = utils.get_logger(__name__)


class PreprocessingPipeline:
    """
    A reusable class to preprocess product data for BERT-based NLP tasks.

    Attributes:
        tokenizer (BertTokenizer): The BERT tokenizer.
        ordered_columns_to_keep (list): List of column names to keep in the cleaned data.

    Methods:
        preprocess(df: DataFrame) -> DataFrame:
            The entire pipeline including cleaning and text processing.
    """

    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        """
        Initializes the preprocessing pipeline with a BERT tokenizer.

        Args:
            bert_model_name (str): Name of the BERT model to use for tokenization.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.ordered_columns_to_keep = [
            "title", "description", "feature", "brand", "asin"]

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """
        Cleans the data.

        Args:
            df (DataFrame): Input dataframe with product data.

        Returns:
            DataFrame: Cleaned dataframe.
        """
        return clean.clean_data(df, self.ordered_columns_to_keep)

    def _process_text(self, df: DataFrame) -> DataFrame:
        """
        Process the text with BERT tokenization.

        Args:
            df (DataFrame): Input dataframe with product data.

        Returns:
            DataFrame: DataFrame with tokenized text column.
        """
        df = transform.combine_text_fields(df)
        return transform.bert_tokenize_text(df, self.tokenizer)

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Preprocess the entire pipeline including cleaning and text processing.

        Args:
            df (DataFrame): Input dataframe with product data.

        Returns:
            DataFrame: Preprocessed dataframe ready for training or inference.
        """
        logger.info("Cleaning the data.")
        cleaned_df = self._clean_data(df)
        logger.info("Processing the text data.")
        processed_df = self._process_text(cleaned_df)
        logger.info("Preprocessing pipeline completed.")
        return processed_df
