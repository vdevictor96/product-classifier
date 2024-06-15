from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType, IntegerType
from transformers import BertTokenizer


def combine_text_fields(df: DataFrame, combined_text_column="combined_text") -> DataFrame:
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

    # Get all column names from the DataFrame
    column_names = df.columns

    # Apply the UDF to combine all columns
    return df.withColumn(combined_text_column, combine_udf(*[col(c) for c in column_names]))


def bert_tokenize_text(df: DataFrame, tokenizer: BertTokenizer, combined_text_column="combined_text") -> DataFrame:
    """
    Tokenizes text using the BERT tokenizer.

    Args:
        df (DataFrame): Input DataFrame.
        tokenizer (BertTokenizer): BERT tokenizer.
        combined_text_column (str, optional): Name of the column with the combined text. Defaults to "combined_text".

    Returns:
        DataFrame: DataFrame with tokenized text column.
    """

    tokenize_udf = udf(lambda text: tokenizer.encode(
        text, add_special_tokens=True, truncation=True, padding='max_length'), ArrayType(IntegerType()))
    return df.withColumn('tokenized_text', tokenize_udf(col(combined_text_column)))
