"""Module that performs the ETL pipeline (Extract, Transform, Load)."""
import fire
import src.data_pipeline.etl.extract as extract
import src.data_pipeline.etl.transform as transform
import src.data_pipeline.etl.load as load
from src.data_pipeline import utils


logger = utils.get_logger(__name__)


def run(input_path: str, output_path: str, fraction=1.0, seed=42):
    """
    Full ETL pipeline including extraction, transformation, and loading.

    Args:
        input_path (str): Path to the input JSONL.gz file.
        output_path (str): Path to save the output Parquet file.
        fraction (float): Fraction of the data to sample. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    logger.info("Starting ETL pipeline.")

    # Step 1: Extract
    logger.info("Extracting data.")
    raw_df = extract.from_jsonl_gz_file(input_path, fraction, seed)

    # Step 2: Transform
    logger.info("Transforming data.")
    transformed_data = transform.preprocess(raw_df)

    # Step 3: Load
    logger.info("Loading data.")
    load.to_parquet_file(transformed_data, output_path)

    logger.info("ETL pipeline completed.")


if __name__ == "__main__":
    fire.Fire(run)
