"""Module that takes care of the extraction of the raw data in an ETL pipeline."""
import gzip
import json
from typing import Generator
from pyspark.sql import SparkSession, DataFrame
import pandas as pd
from src.data_pipeline import utils

logger = utils.get_logger(__name__)


def from_jsonl_gz_file(path, data_fraction=1.0, seed=42) -> DataFrame:
    """Returns a Spark DataFrame from the JSON objects in the gzip file.

    Args:
        path (str): The path to the gzip file.
        data_fraction (float): Fraction of the data to sample. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        DataFrame: A Spark DataFrame containing the JSON objects.
    """
    try:
        spark = SparkSession.builder \
            .appName("ProductClassifier") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "6g") \
            .getOrCreate()

        # Load JSON data
        df = spark.read.json(path)
        if data_fraction < 1.0:
            df = df.sample(fraction=data_fraction, seed=seed)
        return df
    except Exception as e:
        logger.error("Failed to extract data using Spark: %s",
                     e, exc_info=True)
        raise


def from_jsonl_gz_file_pandas(path: str, data_fraction: float = 1.0, seed=42) -> pd.DataFrame:
    """Returns a Pandas DataFrame from the JSON objects in the gzip file.

    Args:
        path (str): The path to the gzip file.
        data_fraction (float): Fraction of the data to sample. Defaults to 1.0.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the JSON objects.
    """
    try:
        # Parse JSON objects from the gzip file
        data_gen = _parse_from_file(path)

        # Convert the generator to a list of JSON objects
        data = list(data_gen)

        # Convert the list of JSON objects to a Pandas DataFrame
        df = pd.DataFrame(data)

        if data_fraction < 1.0:
            df = df.sample(frac=data_fraction, random_state=seed)

        return df
    except Exception as e:
        logger.error("Failed to extract data using Pandas: %s",
                     e, exc_info=True)
        raise


def _parse_from_file(path: str) -> Generator:
    """ Parses the zip file and returns a generator of json objects

    Args:
        path (str): The path to the zip file

    Returns:
        Generator: A generator that yields json objects
    """
    try:
        g = gzip.open(path, "r")
        for l in g:
            yield json.loads(l)
    except Exception as e:
        logger.error("Failed to parse file %s: %s", path, e, exc_info=True)
        raise
