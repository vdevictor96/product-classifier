"""Module for utility functions."""
import gzip
import logging
import random
from io import StringIO

import yaml
import torch
import numpy as np

import pandas as pd
from pyspark.sql import SparkSession, DataFrame




def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO)
    logger_instance = logging.getLogger(name)

    return logger_instance

logger = get_logger(__name__)

def read_jsonl_to_pandas(path: str) -> pd.DataFrame:
    """
    Reads a Parquet file into a Pandas DataFrame.

    Args:
        path (str): The path to the Parquet file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the Parquet file.
    """
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            d = file.read()
            df = pd.read_json(StringIO(d), lines=True, encoding='utf-8')
        return df
    except Exception as e:
        logger.error(
            "Failed to read JSONL to Pandas DataFrame: %s", e, exc_info=True)
        raise


def read_parquet_to_pandas(path: str) -> pd.DataFrame:
    """
    Reads a Parquet file into a Pandas DataFrame.

    Args:
        path (str): The path to the Parquet file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the Parquet file.
    """
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(
            "Failed to read Parquet to Pandas DataFrame: %s", e, exc_info=True)
        raise


def read_jsonl_to_pyspark(path) -> DataFrame:
    """Returns a Spark DataFrame from the JSON objects in the gzip file.

    Args:
        path (str): The path to the gzip file.

    Returns:
        DataFrame: A Spark DataFrame containing the JSON objects.
    """
    try:
        spark = _get_spark_session()
        # Load JSON data
        return spark.read.json(path)
    except Exception as e:
        logger.error(
            "Failed to read JSONL to PySpark DataFrame: %s", e, exc_info=True)
        raise


def read_parquet_to_pyspark(path: str) -> DataFrame:
    """
    Reads a Parquet file into a PySpark DataFrame.

    Args:
        path (str): The path to the Parquet file.

    Returns:
        DataFrame: A PySpark DataFrame containing the data from the Parquet file.
    """
    try:
        spark = _get_spark_session()
        return spark.read.parquet(path)
    except Exception as e:
        logger.error(
            "Failed to read Parquet to PySpark DataFrame: %s", e, exc_info=True)
        raise


def set_seed(seed, device='cuda'):
    """Set the random seed for reproducibility.

    Args:
        seed (int): Random seed.
        device (str): Device to set seed for. Defaults to 'cuda'.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration file from yml file on the specified path."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error("Failed to load configuration file: %s", e, exc_info=True)
        raise





def _get_spark_session():
    """
    Retrieves or creates a Spark session.

    Returns:
        SparkSession: A Spark session.
    """
    try:
        spark = SparkSession.builder \
            .appName("ProductClassifier") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "6g") \
            .getOrCreate()
        # Set log level programmatically
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        logger.error(
            "Failed to create or retrieve Spark session: %s", e, exc_info=True)
        raise
