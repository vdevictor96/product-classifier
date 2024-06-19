"""Module that defines the PyTorch IterableDataset for the product data."""
from typing import Iterator

from torch import Tensor, tensor
from torch.utils.data import IterableDataset
from transformers import BertTokenizer
from pyspark.sql import DataFrame
from sklearn.preprocessing import LabelEncoder


class ProductDataset(IterableDataset):
    """
    PyTorch IterableDataset for product data.

    Attributes:
        df (DataFrame): The Spark DataFrame containing the data.
        label_encoder (LabelEncoder): An encoder for converting category labels to integers.
        max_length (int): Maximum length of the tokens.
    """

    def __init__(self, df: DataFrame, label_encoder: LabelEncoder, max_length=512):
        """
        Initializes the dataset with a Spark DataFrame, a label encoder, and a tokenizer.

        Args:
            df (DataFrame): The Spark DataFrame to load data from.
            label_encoder (LabelEncoder): The encoder for converting category labels to integers.
            max_length (int): The maximum length of the tokenized input.
        """
        self.df = df
        self.label_encoder = label_encoder
        self.max_length = max_length

    def process_row(self, row) -> tuple[Tensor, Tensor, Tensor]:
        """
        Processes a single row of the DataFrame.

        Args:
            row: A row from DataFrame.

        Returns:
            tuple containing input_ids, attention_mask, and label as tensors.
        """
        label = self.label_encoder.transform([row['main_cat']])[0]
        input_ids = tensor(row['input_ids'])
        attention_mask = tensor(row['attention_mask'])
        return input_ids, attention_mask, tensor(label)

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
        return (self.process_row(row) for row in self.df.rdd.toLocalIterator())
