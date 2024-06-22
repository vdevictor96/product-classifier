from collections import defaultdict

import datetime
from pprint import pformat
import pandas as pd
import numpy as np
import mlflow
import fire
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from pyspark.sql import DataFrame
from pyspark.sql.functions import rand

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from src.training.constants import CATEGORIES
from src.training import utils
from src.training.settings import SETTINGS
from src.training.datasets.product_dataset import ProductIterableDataset, ProductDataset


logger = utils.get_logger(__name__)


def create_datasets(df: DataFrame, label_encoder: LabelEncoder, tokenizer: AutoTokenizer, encoded: bool, max_length=512, seed=42) -> tuple[ProductIterableDataset, ProductIterableDataset, ProductIterableDataset]:
    """
    Divides the data into train, validation, and test and creates PyTorch Datasets.

    Args:
        df (DataFrame): DataFrame containing the preprocessed data.
        label_encoder (LabelEncoder): Encoder for the labels.
        tokenizer (AutoTokenizer): BERT tokenizer.
        encoded (bool): Whether the data is already encoded.
        max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[ProductIterableDataset, ProductIterableDataset, ProductIterableDataset]: Train, validation, and test datasets.
    """

    # Create a Spark DataFrame for each data split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=seed)

    # Shuffle the data for better training
    train_df = train_df.orderBy(rand(seed=seed))
    val_df = val_df.orderBy(rand(seed=seed))
    test_df = test_df.orderBy(rand(seed=seed))

    train_dataset = ProductIterableDataset(
        train_df, label_encoder, tokenizer, encoded, max_length)
    val_dataset = ProductIterableDataset(
        val_df, label_encoder, tokenizer, encoded, max_length)
    test_dataset = ProductIterableDataset(
        test_df, label_encoder, tokenizer, encoded, max_length)

    return train_dataset, val_dataset, test_dataset


def create_in_memory_datasets(df: pd.DataFrame, label_encoder: LabelEncoder, tokenizer: AutoTokenizer, encoded: bool, max_length=512, seed=42) -> tuple[ProductDataset, ProductDataset, ProductDataset]:
    """
    Divides the data into train, validation, and test and creates PyTorch Datasets.

    Args:
        df (pd.DataFrame): DataFrame containing the preprocessed data.
        label_encoder (LabelEncoder): Encoder for the labels.
        tokenizer (AutoTokenizer): BERT tokenizer.
        encoded (bool): Whether the data is already encoded.
        max_length (int): Maximum length of the input sequence for BERT Tokenizer. Defaults to 512.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[ProductDataset, ProductDataset, ProductDataset]: Train, validation, and test datasets.
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=seed)

    train_dataset = ProductDataset(
        train_df, label_encoder, tokenizer, encoded, max_length)
    val_dataset = ProductDataset(
        val_df, label_encoder, tokenizer, encoded, max_length)
    test_dataset = ProductDataset(
        test_df, label_encoder, tokenizer, encoded, max_length)

    return train_dataset, val_dataset, test_dataset


def create_loaders(train_dataset: ProductIterableDataset, val_dataset: ProductIterableDataset, test_dataset: ProductIterableDataset, batch_size: int, test_batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        train_dataset (ProductIterableDataset): Training dataset.
        val_dataset (ProductIterableDataset): Validation dataset.
        test_dataset (ProductIterableDataset): Test dataset.
        batch_size (int): Batch size for training and validation.
        test_batch_size (int): Batch size for testing.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
        train_loader_len = len(train_loader)
        val_loader_len = len(val_loader)
        test_loader_len = len(test_loader)

        logger.info("Train loader size: %d.", train_loader_len)
        logger.info("Val loader size: %d.", val_loader_len)
        logger.info("Test loader size: %d.", test_loader_len)
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error("Failed to create DataLoaders: %s", e, exc_info=True)
        raise


def create_in_memory_loaders(train_dataset: ProductDataset, val_dataset: ProductDataset, test_dataset: ProductDataset, batch_size: int, test_batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        train_dataset (ProductDataset): Training dataset.
        val_dataset (ProductDataset): Validation dataset.
        test_dataset (ProductDataset): Test dataset.
        batch_size (int): Batch size for training and validation.
        test_batch_size (int): Batch size for testing.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False)

        logger.info("Train loader size: %d.", len(train_loader))
        logger.info("Val loader size: %d.", len(val_loader))
        logger.info("Test loader size: %d.", len(test_loader))
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error("Failed to create DataLoaders: %s", e, exc_info=True)
        raise


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], log: bool = True):
    """
    Plots and logs the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list[str]): List of class labels.
        log (bool): Whether to log the plot to MLflow. Defaults to True.
    """
    try:
        cm_percentage = cm * 100
        plt.figure(figsize=(20, 12))
        sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap='Blues',
                    annot_kws={"size": 10, "ha": 'center', "va": 'center'})
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        # Adding an offset and aligning vertically
        plt.yticks(tick_marks + 0.5, labels, rotation=0, va='center')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        if log:
            mlflow.log_artifact('confusion_matrix.png')
    except Exception as e:
        logger.error("Failed to plot or log confusion matrix: %s",
                     str(e), exc_info=True)
        raise


def train(loader: DataLoader, model: AutoModelForSequenceClassification, optimizer: AdamW, scheduler: LambdaLR, device: str, epoch: int, num_epochs: int):
    """
    Trains the model for one epoch.

    Args:
        loader (DataLoader): PyTorch DataLoader for training data.
        model (AutoModelForSequenceClassification): BERT model for text classification.
        optimizer (AdamW): Optimizer for training.
        scheduler (LambdaLR): Learning rate scheduler.
        device (str): Device to use for training.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """

    loader_len = len(loader)
    model.train()
    total_loss = 0
    total_train_acc = 0.0
    # Dictionary to count occurrences of each label
    label_counter = defaultdict(int)
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        batch_acc = accuracy_score(label_ids, logits.argmax(axis=1))
        total_train_acc += batch_acc
        # Update label counts
        unique, counts = np.unique(label_ids, return_counts=True)
        for label, count in zip(unique, counts):
            label_counter[label] += count

        if (i+1) % 100 == 0 or (i+1) == loader_len:
            step_loss = total_loss / i
            step_acc = total_train_acc / i
            logger.info(
                'Epoch %d/%d, Step %d/%d, Loss: %.4f, Accuracy: %.2f%%',
                epoch+1, num_epochs, i, loader_len, step_loss, step_acc * 100)
            mlflow.log_metrics(
                {'Step Loss': step_loss, 'Step Accuracy': step_acc}, step=epoch * loader_len + i)
    logger.info("--------------------")
    logger.info("Epoch %d/%d, Average Training Loss: %.6f, Average Training Accuracy: %.2f%%",
                epoch + 1, num_epochs, total_loss / loader_len, total_train_acc / loader_len * 100)
    logger.info("--------------------")
    mlflow.log_metrics({'Training Loss': total_loss / loader_len,
                       'Training Accuracy': total_train_acc / loader_len}, step=epoch+1)


def evaluate(loader: DataLoader, model: AutoModelForSequenceClassification, device: str, epoch: int, num_epochs: int):
    """
    Evaluates the model on the validation set after each training epoch.

    Args:
        loader (DataLoader): PyTorch DataLoader for validation data.
        model (AutoModelForSequenceClassification): BERT model for text classification.
        device (str): Device to use for evaluation.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        float: Validation loss.
    """
    loader_len = len(loader)
    model.eval()
    eval_loss, eval_acc = 0, 0
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", unit="batch"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                eval_acc += accuracy_score(label_ids, logits.argmax(axis=1))

        eval_loss = eval_loss / loader_len
        eval_acc = eval_acc / loader_len
        logger.info("--------------------")
        logger.info(
            'Epoch %d/%d, Validation Loss: %.4f, Validation Accuracy: %.2f%%',
            epoch+1, num_epochs, eval_loss, eval_acc * 100)
        logger.info("--------------------")
        mlflow.log_metrics({'Validation Loss': eval_loss,
                            'Validation Accuracy': eval_acc}, step=epoch+1)
        return eval_loss, eval_acc
    except Exception as e:
        logger.error("Error during model evaluation: %s", e, exc_info=True)
        raise


def test(loader: DataLoader, decoded_categories: list[str], model: AutoModelForSequenceClassification, device: str):
    """
    Tests the model on the test set.

    Args:
        loader (DataLoader): PyTorch DataLoader for test data.
        decoded_categories (list[str]): List of decoded category labels.
        model (AutoModelForSequenceClassification): BERT model for text classification.
        device (str): Device to use for testing.

    Returns:
        dict: Dictionary containing the test metrics.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing", unit="batch"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                all_labels.extend(label_ids.tolist())
                all_predictions.extend(logits.argmax(axis=1).tolist())

        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(
            all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions,
                              average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_predictions)
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(
            cm.astype('float'), cm_sum, where=cm_sum != 0)
        cm_normalized = np.nan_to_num(cm_normalized)
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        logger.info('Test Accuracy: %.2f%%, F1 Score: %.2f, Precision: %.2f, Recall: %.2f',
                    acc * 100, f1, precision, recall)
        test_metrics = {
            'Accuracy': acc,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
        }
        # Log the test metrics
        for key, value in test_metrics.items():
            mlflow.log_metric(f"Test {key}", value)
        # Plot and log the confusion matrix
        plot_confusion_matrix(cm_normalized, decoded_categories)

        # Log the best model
        mlflow.pytorch.log_model(model, "model")
        return test_metrics
    except Exception as e:
        logger.error("Error during model testing: %s", e, exc_info=True)
        raise


def run(config='src/training/configs/default_config.yml', **overrides):
    """
    Function to run the training pipeline.

    Args:
        config (str): Path to the configuration file. Defaults to 'src/training/default_config.yml'.
            input_data_path (str): Path to the input JSONL.gz file.
            input_data_encoded (bool): Whether the data is already encoded. Defaults to False.
            bert_model_name (str): Name of the BERT model to use for tokenization. Defaults to 'bert-base-uncased'.
            data_load ('memory' | 'distributed'): Method to load the data. Defaults to 'distributed'.
            trainable_layers (int | None): Number of layers to keep trainable. None for all layers. Defaults to None.
            num_epochs (int): Number of epochs to train the model. Defaults to 3.
            batch_size (int): Batch size for training and validation. Defaults to 8.
            test_batch_size (int): Batch size for testing. Defaults to 32.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-5.
            optimizer (str): Optimizer to use for training. Defaults to 'AdamW'.
            seed (int): Random seed for reproducibility. Defaults to 42.
            device (str): Device to use for training. Defaults to 'cuda'.
            data_fraction (float): Fraction of the data to sample for training. Defaults to 1.0.
    """
    try:
        logger.info("Logging in to MLflow server hosted at Databricks CE.")
        mlflow.login()
        mlflow.set_tracking_uri(SETTINGS["MLFLOW_TRACKING_URI"])

        logger.info("Setting up MLflow experiment.")
        mlflow.set_experiment(SETTINGS["MLFLOW_EXPERIMENT_NAME"])

        config = utils.load_config(config)
        config.update(overrides)

        input_data_path = config.get(
            "input_data_path", "data/preprocessed/amz_products_small_preprocessed_v1.parquet")
        bert_model_name = config.get("bert_model_name", "bert-base-uncased")
        data_load = config.get("data_load", "distributed")
        input_data_encoded = config.get("input_data_encoded", False)
        trainable_layers = config.get("trainable_layers", None)
        num_epochs = config.get("num_epochs", 3)
        batch_size = config.get("batch_size", 8)
        test_batch_size = config.get("test_batch_size", 32)
        learning_rate = float(config.get("learning_rate", 1e-5))
        optimizer = config.get("optimizer", "AdamW")
        seed = config.get("seed", 42)
        device = config.get("device", torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))
        data_fraction = config.get("data_fraction", 1.0)

        if device == 'cuda':
            logger.info("Checking if CUDA is available.")
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            logger.info("Using device: %s.", device)

        logger.info("Setting random seed: %d.", seed)
        utils.set_seed(seed, device=device)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        with mlflow.start_run(run_name=f"Run {current_time}"):
            params = {
                "input_data_path": input_data_path,
                "bert_model_name": bert_model_name,
                "data_load": data_load,
                "input_data_encoded": str(input_data_encoded),
                "trainable_layers": trainable_layers,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "test_batch_size": test_batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "seed": seed,
                "device": str(device),
                "data_fraction": data_fraction
            }
            mlflow.log_params(params)
            logger.info("Starting run with configuration:")
            logger.info("\n%s", pformat(params))

            # Encode categories
            label_encoder = LabelEncoder()
            label_encoder.fit(CATEGORIES)

            # Create BERT tokenizers
            logger.info("Creating BERT tokenizer for model: %s.",
                        bert_model_name)
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

            logger.info("Creating the BERT model for text classification.")
            # Create the Bert model for text classification
            model = AutoModelForSequenceClassification.from_pretrained(
                bert_model_name, num_labels=len(CATEGORIES))

            model.to(device)

            # Freeze all layers except the last trainable_layers
            if trainable_layers is not None and trainable_layers != 'None':
                trainable_layers = int(trainable_layers)
                logger.info(
                    "Freezing all layers except the last %d layers.", trainable_layers)
                total_params, trainable_params = utils.freeze_layers(
                    model, trainable_layers)
                logger.info("Total model parameters: %d.", total_params)
                logger.info("Total trainable parameters: %d.",
                            trainable_params)
            # Create the optimizer
            optimizer = AdamW(model.parameters(), lr=learning_rate,  eps=1e-8)

            # Load the data
            if data_load == 'distributed':
                logger.info(
                    "Extracting preprocessed data from file to a distributed PySpark DataFrame.")

                df = utils.read_parquet_to_pyspark(input_data_path)
                data_size = df.count()
                logger.info("Data extracted with size: %d rows", data_size)

                if data_fraction < 1.0:
                    logger.info("Sampling data with fraction: %f.",
                                data_fraction)
                    df = df.sample(fraction=data_fraction, seed=seed)
                    data_size = df.count()
                    logger.info("Data sampled with size: %d rows", data_size)

                logger.info("Creating training and validation datasets.")
                train_dataset, val_dataset, test_dataset = create_datasets(
                    df, label_encoder, tokenizer, input_data_encoded, 512, seed=seed)

                logger.info(
                    "Creating the train and validation dataloaders with batch size: %d.", batch_size)
                logger.info(
                    "Creating the test dataloader with batch size: %d.", test_batch_size)
                # Create the dataloaders
                train_loader, val_loader, test_loader = create_loaders(
                    train_dataset, val_dataset, test_dataset, batch_size, test_batch_size)
            else:
                logger.info(
                    "Extracting preprocessed data from file to an in memory Pandas DataFrame.")
                df = utils.read_parquet_to_pandas(input_data_path)
                data_size = len(df)
                logger.info("Data extracted with size: %d rows", data_size)

                if data_fraction < 1.0:
                    logger.info("Sampling data with fraction: %f.",
                                data_fraction)
                    df = df.sample(frac=data_fraction, random_state=seed)
                    data_size = len(df)
                    logger.info("Data sampled with size: %d rows", data_size)

                logger.info("Creating training and validation datasets.")
                train_dataset, val_dataset, test_dataset = create_in_memory_datasets(
                    df, label_encoder, tokenizer, input_data_encoded, 512, seed=seed)

                logger.info(
                    "Creating the train and validation dataloaders with batch size: %d.", batch_size)
                logger.info(
                    "Creating the test dataloader with batch size: %d.", test_batch_size)
                # Create the dataloaders
                train_loader, val_loader, test_loader = create_in_memory_loaders(
                    train_dataset, val_dataset, test_dataset, batch_size, test_batch_size)

            # Create the learning rate scheduler
            total_steps = len(train_loader) * num_epochs
            # Set the warmup steps to 10% of the total steps
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            logger.info("Training the model.")

            best_model_state = model.state_dict()
            best_eval_loss = float('inf')
            try:
                # Training loop
                for epoch in range(num_epochs):
                    # Train one epoch
                    # Shuffle the data after the first epoch if using IterableDataset
                    if (data_load == 'distributed' and epoch > 0):
                        train_loader.dataset.reshuffle(seed=seed)
                        train_loader = DataLoader(
                            train_loader.dataset, batch_size=batch_size)

                    train(train_loader, model, optimizer, scheduler,
                          device, epoch, num_epochs)
                    # Validation after each epoch
                    eval_loss, eval_acc = evaluate(
                        val_loader, model, device, epoch, num_epochs)

                    # Save the model if it has the best validation loss so far
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_state = model.state_dict()
                        logger.info(
                            "New best model saved in epoch %d with validation loss: %.4f and accuracy: %.2f%%",
                            epoch+1, eval_loss, eval_acc * 100)

                # Load the best model before testing
                model.load_state_dict(best_model_state)
                logger.info("Best model loaded for testing.")
                # Evaluate the best model on test data
                encoded_categories = label_encoder.transform(CATEGORIES)
                decoded_categories = label_encoder.inverse_transform(
                    encoded_categories)
                test(test_loader, decoded_categories, model, device)
            except Exception as e:
                logger.error("Error during model training: %s",
                             e, exc_info=True)
                raise

    except Exception as e:
        logger.error(
            "An error occurred during the training pipeline: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    fire.Fire(run)
