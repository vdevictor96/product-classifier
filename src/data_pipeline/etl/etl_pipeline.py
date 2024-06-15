import fire
from src.data_pipeline.etl.extract import from_jsonl_gz_file
from src.data_pipeline.etl.load import to_parquet_file

from src.data_pipeline import utils
from src.data_pipeline.preprocessing_pipeline import PreprocessingPipeline


logger = utils.get_logger(__name__)


def run(input_path: str, output_path: str, fraction=1.0):
    """
    Full ETL pipeline including extraction, transformation, and loading.

    Args:
        input_path (str): Path to the input JSONL.gz file.
        output_path (str): Path to save the output Parquet file.
        fraction (float): Fraction of the data to sample. Defaults to 1.
    """
    logger.info("Starting ETL pipeline.")

    # Step 1: Extract
    logger.info("Extracting data.")
    raw_df = from_jsonl_gz_file(input_path, fraction)

    # Step 2: Transform
    logger.info("Transforming data.")
    preprocessing_pipeline = PreprocessingPipeline()
    transformed_data = preprocessing_pipeline.preprocess(raw_df)

    # Step 3: Load
    logger.info("Loading data.")
    to_parquet_file(transformed_data, output_path)

    logger.info("ETL pipeline completed.")


if __name__ == "__main__":
    fire.Fire(run)
