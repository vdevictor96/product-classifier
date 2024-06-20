"""Module for utility functions."""
import logging
from src.inference.api.schemas.product_descriptions import ProductDescription


def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    return logger



def preprocess_data(data: ProductDescription) -> str:
    """ 
    Preprocesses the input data for the model.

    Args:
        data (ProductDescription): The input data to preprocess.

    Returns:
        str: The preprocessed text data.
    """
    combined_text = ". ".join(
        [data.title] + data.description + data.feature + [data.brand])
    return combined_text
