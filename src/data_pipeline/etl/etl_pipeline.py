"""Module that performs the ETL pipeline (Extract, Transform, Load)."""
import fire
from src.data_pipeline.etl.extract import from_jsonl_gz_file
from src.data_pipeline.etl.load import to_parquet_file

from src.data_pipeline import utils
from src.data_pipeline.preprocessing_pipeline import PreprocessingPipeline


logger = utils.get_logger(__name__)


def run(input_path: str, output_path: str, bert_model_name = "bert-base-uncased", fraction = 1.0, seed = 42):
    """
    Full ETL pipeline including extraction, transformation, and loading.

    Args:
        input_path (str): Path to the input JSONL.gz file.
        output_path (str): Path to save the output Parquet file.
        bert_model_name (str): Name of the BERT model to use for tokenization. Defaults to "bert-base-uncased".
        fraction (float): Fraction of the data to sample. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    logger.info("Starting ETL pipeline.")

    # Step 1: Extract
    logger.info("Extracting data.")
    raw_df = from_jsonl_gz_file(input_path, fraction, seed)

    # Step 2: Transform
    logger.info("Transforming data.")
    preprocessing_pipeline = PreprocessingPipeline(bert_model_name=bert_model_name)
    transformed_data = preprocessing_pipeline.preprocess(raw_df)

    # Step 3: Load
    logger.info("Loading data.")
    to_parquet_file(transformed_data, output_path)

    logger.info("ETL pipeline completed.")


if __name__ == "__main__":
    fire.Fire(run)
