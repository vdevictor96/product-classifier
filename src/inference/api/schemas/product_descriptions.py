"""Module for Pydantic schema used in the API for input data."""
from pydantic import BaseModel


class ProductDescription(BaseModel):
    """Pydantic model for the expected input product description.
    
    Attributes:
        also_buy (list[str]): List of also bought products.
        also_view (list[str]): List of also viewed products.
        asin (str): Amazon Standard Identification Number.
        brand (str): Brand of the product.
        category (list[str]): List of categories the product belongs to.
        description (list[str]): List of product descriptions.
        feature (list[str]): List of product features.
        image (list[str]): List of image URLs.
        price (str): Price of the product.
        title (str): Title of the product.
    """
    also_buy: list[str] 
    also_view: list[str]
    asin: str
    brand: str
    category: list[str]
    description: list[str]
    feature: list[str]
    image: list[str]
    price: str
    title: str
