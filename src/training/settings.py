"""Module for setting up the environment."""
import os
from dotenv import load_dotenv


def load_env_vars() -> dict[str]:
    """
    Load environment variables from .env file.
    
    Returns:
        dict[str]: Environment variables.

    """
    load_dotenv("src/training/.env")
    return dict(os.environ)

SETTINGS = load_env_vars()
