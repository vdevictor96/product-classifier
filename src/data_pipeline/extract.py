import gzip
import json
from typing import Generator
from pyspark.sql import SparkSession, DataFrame


def from_file(path, fraction=1) -> DataFrame:
    """Returns a Spark DataFrame from the JSON objects in the gzip file.

    Args:
        path (str): The path to the gzip file.
        fraction (float): Fraction of the data to sample. Defaults to 1.

    Returns:
        DataFrame: A Spark DataFrame containing the JSON objects.
    """
    spark = SparkSession.builder \
        .appName("JSONReader") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # Load JSON data
    df = spark.read.json(path)
    return df.sample(fraction)


def _parse_from_file(path: str) -> Generator:
    """ Parses the zip file and returns a generator of json objects

    Args:
        path (str): The path to the zip file

    Returns:
        Generator: A generator that yields json objects
    """

    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)
