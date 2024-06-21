"""Module that defines the PyTorch IterableDataset for the product data."""
from typing import Iterator

from torch import Tensor, tensor
from torch.utils.data import IterableDataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
from pyspark.sql import DataFrame
from sklearn.preprocessing import LabelEncoder
from src.training import utils

logger = utils.get_logger(__name__)


class ProductIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for product data.

    Attributes:
        df (DataFrame): The Spark DataFrame containing the data.
        label_encoder (LabelEncoder): An encoder for converting category labels to integers.
        tokenizer (BertTokenizer): The BERT tokenizer to use.
        encoded (bool): Whether the data is already encoded or not.
        max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
    """

    def __init__(self, df: DataFrame, label_encoder: LabelEncoder, tokenizer: AutoTokenizer, encoded: bool, max_length=512):
        """
        Initializes the dataset with a Spark DataFrame, a label encoder, and a tokenizer.

        Args:
            df (DataFrame): The Spark DataFrame to load data from.
            label_encoder (LabelEncoder): The encoder for converting category labels to integers.
            tokenizer (BertTokenizer): The BERT tokenizer to use.
            encoded (bool): Whether the data is already encoded or not.
            max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
        """
        self.df = df
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.encoded = encoded
        self.max_length = max_length

    def process_row(self, row) -> tuple[Tensor, Tensor, Tensor]:
        """
        Processes a single row of the DataFrame.

        Args:
            row: A row from DataFrame.

        Returns:
            tuple containing input_ids, attention_mask, and label as tensors.
        """
        try:
            label = self.label_encoder.transform([row['main_cat']])[0]
            encoded_input = self.tokenizer.encode_plus(
                text=row['combined_text'],
                add_special_tokens=True,
                max_length=self.max_length,
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_input['input_ids'].squeeze(0)
            attention_mask = encoded_input['attention_mask'].squeeze(0)
            return input_ids, attention_mask, tensor(label)
        except Exception as e:
            logger.error("Failed to process row: %s", e, exc_info=True)
            raise

    def process_encoded_row(self, row) -> tuple[Tensor, Tensor, Tensor]:
        """
        Processes a single row of the DataFrame that is already encoded.

        Args:
            row: A row from DataFrame.

        Returns:
            tuple containing input_ids, attention_mask, and label as tensors.
        """
        try:
            input_ids = tensor(row['input_ids'])
            attention_mask = tensor(row['attention_mask'])
            label = self.label_encoder.transform([row['main_cat']])[0]
            return input_ids, attention_mask, label
        except Exception as e:
            logger.error("Failed to process row: %s", e, exc_info=True)
            raise

    def __len__(self):
        return self.df.count()

    # def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    #     """
    #     Iterator that yields batches of data using DataFrame operations.
    #     Note: This version is faster but requires more memory.

    #     Yields:
    #         A batch of data as a tuple of input_ids, attention_mask, and labels.
    #     """
    #     pandas_df = self.df.toPandas()
    #     return (self.process_row(row) for index, row in pandas_df.iterrows())

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """
        Iterator that yields batches of data.
        Note: This version is slower but requires less memory.

        Yields:
            A batch of data as a tuple of input_ids, attention_mask, and labels.
        """
        if self.encoded:
            return (self.process_encoded_row(row) for row in self.df.rdd.toLocalIterator())
        return (self.process_row(row) for row in self.df.rdd.toLocalIterator())


class ProductDataset(Dataset):
    """
    PyTorch Dataset for product data.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        label_encoder (LabelEncoder): An encoder for converting category labels to integers.
        tokenizer (AutoTokenizer): The BERT tokenizer to use.
        encoded (bool): Whether the data is already encoded or not.
        max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
    """

    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, tokenizer: AutoTokenizer, encoded: bool, max_length=512):
        """
        Initializes the dataset with a pandas DataFrame, a label encoder, and a tokenizer.

        Args:
            df (pd.DataFrame): The pandas DataFrame to load data from.
            label_encoder (LabelEncoder): The encoder for converting category labels to integers.
            tokenizer (AutoTokenizer): The BERT tokenizer to use.
            encoded (bool): Whether the data is already encoded or not.
            max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
        """
        self.df = df
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.encoded = encoded
        self.max_length = max_length

    def process_row(self, row) -> tuple[tensor, tensor, tensor]:
        """
        Processes a single row of the DataFrame.

        Args:
            row: A row from DataFrame.

        Returns:
            tuple containing input_ids, attention_mask, and label as tensors.
        """
        try:
            label = self.label_encoder.transform([row['main_cat']])[0]
            encoded_input = self.tokenizer.encode_plus(
                text=row['combined_text'],
                add_special_tokens=True,
                max_length=self.max_length,
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_input['input_ids'].squeeze(0)
            attention_mask = encoded_input['attention_mask'].squeeze(0)
            return input_ids, attention_mask, tensor(label)
        except Exception as e:
            logger.error("Failed to process row: %s", e, exc_info=True)
            raise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[tensor, tensor, tensor]:
        """
        Retrieves a single data point by index.

        Args:
            idx: Index of the data point.

        Returns:
            tuple containing input_ids, attention_mask, and label as tensors.
        """
        row = self.df.iloc[idx]
        if self.encoded:
            label = self.label_encoder.transform([row['main_cat']])[0]
            return tensor(row['input_ids']), tensor(row['attention_mask']), tensor(label)
        return self.process_row(row)
