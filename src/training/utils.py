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


def read_parquet_to_pandas(path: str, use_pyspark=False) -> pd.DataFrame:
    """
    Reads a Parquet file into a Pandas DataFrame.

    Args:
        path (str): The path to the Parquet file.
        use_pyspark (bool): Whether to use PySpark to read the Parquet file. Defaults to False.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the Parquet file.
    """
    try:
        if use_pyspark:
            return read_parquet_to_pyspark(path).toPandas()
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


def freeze_layers(model, layers_count):
    """
    Freezes all layers except the last layers_count layers.

    Args:
        model: PyTorch model.
        layers_count: Number of layers to keep trainable.

    Returns:
        int: The number of trainable parameters.
        list: List of layers that are trainable.
    """
    layers = []
    # Retrieve the last layers keys
    if hasattr(model, 'classifier'):
        layers.append(('classifier', model.classifier))
        layers_count -= 1
    if layers_count > 0 and hasattr(model.bert, 'pooler'):
        layers.append(('bert.pooler', model.bert.pooler))
        layers_count -= 1
    if layers_count > 0:
        total_encoder_layers = len(model.bert.encoder.layer)
        encoder_layers_to_add = min(
            layers_count, total_encoder_layers)
        start_index = total_encoder_layers - encoder_layers_to_add
        layers.extend([
            (f'bert.encoder.layer.{i}', layer)
            for i, layer in enumerate(model.bert.encoder.layer[-encoder_layers_to_add:], start=start_index)
        ])
    trainable_params = 0
    # Set requires_grad to False for all layers except the last ones
    for param in model.parameters():
        param.requires_grad = False
    for _, layer in layers:
        for param in layer.parameters():
            param.requires_grad = True
            trainable_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    return total_params, trainable_params


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
