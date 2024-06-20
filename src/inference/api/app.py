"""This module contains the FastAPI application and its routes."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer
import torch
from src.inference.api.constants import CATEGORIES
from src.inference.api.schemas.product_descriptions import ProductDescription
from src.inference.api.schemas.predictions import PredictionResponse
from src.inference import utils
from src.inference.settings import SETTINGS

logger = utils.get_logger(__name__)


ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application.
    """

    logger.info("Logging in to MLflow server hosted at Databricks CE.")
    mlflow.login()
    mlflow.set_tracking_uri(SETTINGS["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(SETTINGS["MLFLOW_EXPERIMENT_NAME"])

    # Get the parameter 'bert_model_name' from the run
    run_id = SETTINGS['MLFLOW_RUN_ID']

    # Loads the PyTorch model from MLflow.
    logger.info("Loading the PyTorch model from MLflow.")
    model_path = f"runs:/{run_id}/model"
    try:
        model = mlflow.pytorch.load_model(
            model_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise ValueError(
            f"Failed to load the model from MLflow: {str(e)}") from e

    # Get run details
    run = mlflow.get_run(run_id)
    bert_model_name = run.data.params.get("bert_model_name")
    if not bert_model_name:
        raise ValueError(
            "BERT model name is not specified in the run parameters or tags.")
        # Get run details
    logger.info("Loaded BERT model name: %s", bert_model_name)

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    ml_artifacts["model"] = model
    ml_artifacts["tokenizer"] = tokenizer
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(product_data: ProductDescription):
    """
    Predict the main category of a product based on its description.

    Args:
        product_data (ProductDescription): The input product data.

    """

    try:
        # Encode the categories
        # label_encoder could be loaded from Mlflow as an artifact attached to the run
        label_encoder = LabelEncoder()
        label_encoder.fit(CATEGORIES)
        # Retrieve the model and tokenizer from the ml_artifacts dictionary
        model = ml_artifacts["model"]
        tokenizer = ml_artifacts["tokenizer"]
        # Preprocess the input data
        preprocessed_text = utils.preprocess_data(product_data)
        # encode the preprocessed text
        encoded_input = tokenizer.encode_plus(
            preprocessed_text,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        # Run the model prediction
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction = torch.argmax(outputs.logits, dim=1)
        return PredictionResponse(main_cat=label_encoder.inverse_transform([prediction.item()])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
