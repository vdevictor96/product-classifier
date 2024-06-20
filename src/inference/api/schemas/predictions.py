"""Module for Pydantic schema used in the API for predictions response."""
from pydantic import BaseModel
from enum import Enum
from src.inference.api.constants import CATEGORIES

CategoryEnum = Enum('CategoryEnum', {c.replace('&', 'and').replace(', ', '_').replace(' ', '_'): c for c in CATEGORIES})

class PredictionResponse(BaseModel):
    """
    Pydantic schema for the predictions response.
    
    Attributes:
        main_cat (CategoryEnum): The predicted category.
    """
    main_cat: CategoryEnum
