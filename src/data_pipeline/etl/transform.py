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

    # Get all column names from the DataFrame except the label column "main_cat"
    column_names = [c for c in df.columns if c != "main_cat"]

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
        DataFrame: DataFrame with input_ids and attention_mask columns.
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

    return df
