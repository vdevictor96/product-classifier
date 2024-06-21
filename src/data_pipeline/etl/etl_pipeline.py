"""Module that performs the ETL pipeline (Extract, Transform, Load)."""
import fire
from transformers import AutoTokenizer
import src.data_pipeline.etl.extract as extract
import src.data_pipeline.etl.transform as transform
import src.data_pipeline.etl.load as load
from src.data_pipeline import utils


logger = utils.get_logger(__name__)


def run(input_path: str, output_path: str, tokenizer=None, data_fraction=1.0, seed=42):
    """
    Full ETL pipeline including extraction, transformation, and loading.

    Args:
        input_path (str): Path to the input JSONL.gz file.
        output_path (str): Path to save the output Parquet file.
        tokenizer (str | None): BERT tokenizer. Defaults to None.
        data_fraction (float): Fraction of the data to sample. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    try:
        logger.info("Starting ETL pipeline.")
        bert_tokenizer = None
        if tokenizer is not None:
            bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Step 1: Extract
        logger.info("Extracting data.")
        raw_df = extract.from_jsonl_gz_file(input_path, data_fraction, seed)

        # Step 2: Transform
        logger.info("Transforming data.")
        transformed_data = transform.preprocess(raw_df, bert_tokenizer)

        # Step 3: Load
        logger.info("Loading data.")
        load.to_parquet_file(transformed_data, output_path)

        logger.info("ETL pipeline completed.")
    except Exception as e:
        logger.error("ETL pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    fire.Fire(run)
