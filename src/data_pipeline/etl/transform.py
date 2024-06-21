"""Module that takes care of preprocessing the raw data in an ETL pipeline."""
from transformers import AutoTokenizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, IntegerType
from src.data_pipeline.etl.preprocessing_pipeline import PreprocessingPipeline
from src.data_pipeline import utils

logger = utils.get_logger(__name__)


def bert_tokenize_text(df: DataFrame, tokenizer: AutoTokenizer, combined_text_column="combined_text") -> DataFrame:
    """
    Tokenizes text using the BERT tokenizer.

    Args:
        df (DataFrame): Input DataFrame.
        tokenizer (AutoTokenizer): BERT tokenizer.
        combined_text_column (str, optional): Name of the column with the combined text. Defaults to "combined_text".

    Returns:
        DataFrame: DataFrame with tokenized text column.
    """

    def tokenize(text):
        # Using tokenizer's encode_plus method to get both input_ids and attention_mask
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        return (encoding['input_ids'], encoding['attention_mask'])

    # Define the UDF with return type as two separate arrays of integers
    tokenize_udf = udf(tokenize, ArrayType(ArrayType(IntegerType())))

    # Apply the UDF to the DataFrame and split the result into two columns
    df = df.withColumn("tokens", tokenize_udf(col(combined_text_column)))
    df = df.withColumn("input_ids", df["tokens"].getItem(0))
    df = df.withColumn("attention_mask", df["tokens"].getItem(1))

    # Drop the intermediate 'tokens' column
    df = df.drop("tokens")
    # Drop the original preprocessed text column
    df = df.drop(combined_text_column)

    return df


def preprocess(df: DataFrame, tokenizer: AutoTokenizer | None) -> DataFrame:
    """
    Preprocesses the raw data for BERT-based NLP tasks.

    Args:
        df (DataFrame): Input DataFrame with raw data.
        tokenizer (AutoTokenizer | None): BERT tokenizer. Defaults to None.

    Returns:
        DataFrame: DataFrame with preprocessed data.
    """
    try:
        preprocessing_pipeline = PreprocessingPipeline()
        preprocessed_df = preprocessing_pipeline.preprocess(df)
        if tokenizer is not None:
            logger.info("Tokenizing text data.")
            return bert_tokenize_text(preprocessed_df, tokenizer)
        return preprocessed_df
    except Exception as e:
        logger.error("Failed to preprocess data: %s", e, exc_info=True)
        raise
