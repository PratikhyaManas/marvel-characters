"""
Process Data Script

This script loads and processes the Marvel Characters dataset for model training.
It performs preprocessing, feature engineering, train/test splitting, and saves
the processed data to the Unity Catalog.

Usage:
    python process_data.py --root_path <path> --env <environment>

Arguments:
    --root_path: Root path to the project files
    --env: Environment name (dev, staging, prod)
"""

import argparse
import sys

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Process Marvel Characters dataset for model training"
    )
    parser.add_argument(
        "--root_path",
        action="store",
        type=str,
        required=True,
        help="Root path to the project files"
    )
    parser.add_argument(
        "--env",
        action="store",
        type=str,
        required=True,
        choices=["dev", "staging", "prod"],
        help="Environment name (dev, staging, prod)"
    )
    
    return parser.parse_args()


def load_data(spark: SparkSession, table_name: str):
    """
    Load Marvel characters dataset from Unity Catalog.
    
    Args:
        spark: Spark session instance
        table_name: Fully qualified table name (catalog.schema.table)
        
    Returns:
        Pandas DataFrame with the loaded data
    """
    try:
        logger.info(f"Loading data from table: {table_name}")
        df = spark.table(table_name).toPandas()
        logger.info(f"✓ Data loaded successfully: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {table_name}: {e}")
        raise


def process_data(config_path: str, env: str) -> None:
    """
    Process the Marvel Characters dataset.
    
    Args:
        config_path: Path to the project configuration YAML file
        env: Environment name (dev, staging, prod)
    """
    logger.info("=" * 80)
    logger.info("Marvel Characters Data Processing")
    logger.info("=" * 80)
    
    try:
        # Load project configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = ProjectConfig.from_yaml(config_path=config_path, env=env)
        logger.info(f"✓ Configuration loaded for environment: {env}")
        
        logger.info("\nConfiguration details:")
        logger.info(yaml.dump(config, default_flow_style=False))
        
        # Initialize Spark session
        logger.info("Initializing Spark session...")
        spark = SparkSession.builder.getOrCreate()
        logger.info(f"✓ Spark session initialized: {spark.version}")
        
        # Load Marvel characters dataset
        marvel_table = f"{config.catalog_name}.{config.schema_name}.marvel_characters"
        logger.info(f"Loading Marvel characters data from: {marvel_table}")
        
        df = load_data(spark, marvel_table)
        new_data = df.copy()
        logger.info("✓ Marvel data loaded for processing")
        
        # Initialize DataProcessor
        logger.info("\nInitializing DataProcessor...")
        data_processor = DataProcessor(new_data, config, spark)
        logger.info("✓ DataProcessor initialized")
        
        # Preprocess the data
        logger.info("\nPreprocessing data...")
        logger.info("Steps: Handling missing values, feature engineering, encoding")
        data_processor.preprocess()
        logger.info("✓ Data preprocessing completed")
        
        # Split the data
        logger.info("\nSplitting data into train and test sets...")
        X_train, X_test = data_processor.split_data()
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Train/Test ratio: {len(X_train)}/{len(X_test)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%/{len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        logger.info("✓ Data splitting completed")
        
        # Save to catalog
        logger.info("\nSaving processed data to Unity Catalog...")
        logger.info(f"Target catalog: {config.catalog_name}.{config.schema_name}")
        data_processor.save_to_catalog(X_train, X_test)
        logger.info("✓ Data saved to catalog successfully")
        
        logger.info("=" * 80)
        logger.info("Data Processing Completed Successfully")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info(f"  - Source table: {marvel_table}")
        logger.info(f"  - Training records: {len(X_train)}")
        logger.info(f"  - Test records: {len(X_test)}")
        logger.info(f"  - Features: {X_train.shape[1]}")
        logger.info(f"  - Target catalog: {config.catalog_name}.{config.schema_name}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Data Processing Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


def main() -> None:
    """Main execution function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        logger.info(f"Starting data processing with root_path={args.root_path}, env={args.env}")
        
        # Build configuration path
        config_path = f"{args.root_path}/files/project_config_marvel.yml"
        
        # Process the data
        process_data(config_path=config_path, env=args.env)
        
        sys.exit(0)
        
    except Exception as e:
        logger.exception(f"Data processing script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
