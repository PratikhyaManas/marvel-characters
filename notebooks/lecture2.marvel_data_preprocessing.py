# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

"""Marvel Characters Data Preprocessing Pipeline.

This notebook handles data loading, preprocessing, splitting, and saving to Databricks catalog.
Optimized for performance and reliability with proper error handling and logging.
"""

import time
from pathlib import Path

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

# Track total execution time
pipeline_start_time = time.time()

# Track total execution time
pipeline_start_time = time.time()

# COMMAND ----------
# Configuration Setup

logger.info("="*60)
logger.info("MARVEL CHARACTERS DATA PREPROCESSING PIPELINE")
logger.info("="*60)

config_start = time.time()

try:
    config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
    logger.info("✓ Configuration loaded successfully")
    logger.info("  Environment: dev")
    logger.info(f"  Catalog: {config.catalog_name}")
    logger.info(f"  Schema: {config.schema_name}")
    logger.info(f"  Target: {config.target}")
    logger.info(f"  Features: {len(config.num_features)} numerical, {len(config.cat_features)} categorical")
    
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

config_time = time.time() - config_start
logger.info(f"Configuration setup time: {config_time:.2f}s\n")

# COMMAND ----------
# Data Loading and Initial Analysis

logger.info("STEP 1: Data Loading")
logger.info("-" * 60)

data_load_start = time.time()
filepath = Path("../data/marvel_characters_dataset.csv")

# Initialize Spark session (reuse existing if available)
spark = SparkSession.builder.appName("MarvelDataPreprocessing").getOrCreate()
logger.info(f"✓ Spark session initialized (version {spark.version})")

try:
    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data using pandas for initial exploration
    df = pd.read_csv(str(filepath))
    
    # Validate dataset
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Display basic info
    logger.info(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  File size: {filepath.stat().st_size / 1024**2:.2f} MB")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {', '.join(df.columns)}")
    
    # Check for target column
    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found in dataset")
    
    # Data quality metrics
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
    
    logger.info(f"  Missing values: {missing_count:,} ({missing_pct:.2f}%)")
    logger.info(f"  Duplicate rows: {df.duplicated().sum():,}")
    
    logger.info(f"\n  Target '{config.target}' distribution:")
    for category, count in df[config.target].value_counts().items():
        logger.info(f"    {category}: {count:,} ({count/len(df)*100:.1f}%)")
    
except FileNotFoundError as e:
    logger.error(f"Data file not found: {filepath}")
    raise
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

data_load_time = time.time() - data_load_start
logger.info(f"\nData loading time: {data_load_time:.2f}s\n")

# COMMAND ----------
# Data Preprocessing

logger.info("STEP 2: Data Preprocessing")
logger.info("-" * 60)

preprocess_start = time.time()

logger.info("Starting data preprocessing...")

try:
    data_processor = DataProcessor(df, config, spark)
    
    # Store initial row count
    rows_before = df.shape[0]
    
    # Preprocess the data
    data_processor.preprocess()
    
    rows_after = data_processor.df.shape[0]
    rows_removed = rows_before - rows_after
    removal_pct = (rows_removed / rows_before) * 100
    
    logger.info("✓ Data preprocessing completed successfully")
    logger.info(f"  Rows before: {rows_before:,}")
    logger.info(f"  Rows after: {rows_after:,}")
    logger.info(f"  Rows removed: {rows_removed:,} ({removal_pct:.2f}%)")
    logger.info(f"  Final columns: {data_processor.df.shape[1]}")
    
    # Memory optimization check
    memory_after = data_processor.df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory usage: {memory_after:.2f} MB")
    
except Exception as e:
    logger.error(f"Error during preprocessing: {e}")
    raise

preprocess_time = time.time() - preprocess_start
logger.info(f"\nPreprocessing time: {preprocess_time:.2f}s\n")

preprocess_time = time.time() - preprocess_start
logger.info(f"\nPreprocessing time: {preprocess_time:.2f}s\n")

# COMMAND ----------
# Data Splitting

logger.info("STEP 3: Data Splitting")
logger.info("-" * 60)

split_start = time.time()

logger.info("Splitting data into train and test sets...")

try:
    X_train, X_test = data_processor.split_data(test_size=0.2, random_state=42)
    
    # Validate splits
    if X_train.empty or X_test.empty:
        raise ValueError("Train or test set is empty after split")
    
    total_rows = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total_rows
    test_ratio = len(X_test) / total_rows
    
    logger.info("✓ Data split completed successfully")
    logger.info(f"  Training set: {X_train.shape} ({train_ratio:.1%})")
    logger.info(f"  Test set: {X_test.shape} ({test_ratio:.1%})")
    
    # Check target distribution balance
    logger.info("\n  Target distribution in train:")
    for category, count in X_train[config.target].value_counts().items():
        logger.info(f"    {category}: {count:,} ({count/len(X_train)*100:.1f}%)")
    
    logger.info("\n  Target distribution in test:")
    for category, count in X_test[config.target].value_counts().items():
        logger.info(f"    {category}: {count:,} ({count/len(X_test)*100:.1f}%)")
    
except Exception as e:
    logger.error(f"Error during data splitting: {e}")
    raise

split_time = time.time() - split_start
logger.info(f"\nData splitting time: {split_time:.2f}s\n")

split_time = time.time() - split_start
logger.info(f"\nData splitting time: {split_time:.2f}s\n")

# COMMAND ----------
# Save to Catalog

logger.info("STEP 4: Save to Databricks Catalog")
logger.info("-" * 60)

save_start = time.time()

logger.info("Saving processed data to Databricks catalog...")

try:
    # Save train and test sets
    data_processor.save_to_catalog(X_train, X_test)
    logger.info("✓ Data saved successfully")
    logger.info(f"  Catalog: {config.catalog_name}")
    logger.info(f"  Schema: {config.schema_name}")
    logger.info("  Tables: train_set, test_set")
    
    # Enable change data feed for tracking changes
    logger.info("\nEnabling change data feed...")
    data_processor.enable_change_data_feed()
    logger.info("✓ Change data feed enabled successfully")
    
except Exception as e:
    logger.error(f"Error saving data to catalog: {e}")
    raise

save_time = time.time() - save_start
logger.info(f"\nSave to catalog time: {save_time:.2f}s\n")

# COMMAND ----------
# Pipeline Summary

pipeline_total_time = time.time() - pipeline_start_time

logger.info("="*60)
logger.info("PIPELINE EXECUTION SUMMARY")
logger.info("="*60)
logger.info("✓ Data preprocessing pipeline completed successfully!")
logger.info("")
logger.info("Execution Times:")
logger.info(f"  Configuration:   {config_time:.2f}s")
logger.info(f"  Data Loading:    {data_load_time:.2f}s")
logger.info(f"  Preprocessing:   {preprocess_time:.2f}s")
logger.info(f"  Data Splitting:  {split_time:.2f}s")
logger.info(f"  Save to Catalog: {save_time:.2f}s")
logger.info(f"  {'─'*25}")
logger.info(f"  Total Time:      {pipeline_total_time:.2f}s")
logger.info("")
logger.info("Data Summary:")
logger.info(f"  Input rows:      {rows_before:,}")
logger.info(f"  Output rows:     {rows_after:,}")
logger.info(f"  Training rows:   {len(X_train):,}")
logger.info(f"  Test rows:       {len(X_test):,}")
logger.info(f"  Features:        {X_train.shape[1]}")
logger.info("")
logger.info(f"Output Location:  {config.catalog_name}.{config.schema_name}")
logger.info("="*60)

# COMMAND ----------