# Databricks notebook source

"""Basic Model Training and Registration Pipeline.

This notebook demonstrates:
- Training a LightGBM classifier for Marvel character survival prediction
- Logging the model with MLflow
- Evaluating model performance
- Registering the model to Unity Catalog
- Retrieving and analyzing model metadata
"""

import json
import os
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, f1_score

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.
    
    :return: True if running in Databricks, False otherwise
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def setup_mlflow_tracking() -> None:
    """Configure MLflow tracking URI based on environment."""
    if not is_databricks():
        load_dotenv()
        profile = os.environ.get("PROFILE", "DEFAULT")
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
        logger.info(f"MLflow configured for profile: {profile}")
    else:
        logger.info("Running in Databricks environment")

# COMMAND ----------
# Environment and Configuration Setup

logger.info("="*60)
logger.info("BASIC MODEL TRAINING & REGISTRATION PIPELINE")
logger.info("="*60)

pipeline_start = time.time()

# Setup MLflow tracking
setup_mlflow_tracking()

# Load configuration
try:
    config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
    logger.info("✓ Configuration loaded successfully")
    logger.info(f"  Catalog: {config.catalog_name}")
    logger.info(f"  Schema: {config.schema_name}")
    logger.info(f"  Target: {config.target}")
    logger.info(f"  Experiment: {config.experiment_name_basic}")
    logger.info(f"  Features: {len(config.num_features)} numerical, {len(config.cat_features)} categorical")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Initialize Spark session
spark = SparkSession.builder.appName("MarvelBasicModelTraining").getOrCreate()
logger.info(f"✓ Spark session initialized (version {spark.version})")

# Setup tags for tracking
tags = Tags(git_sha="abcd12345", branch="main")
logger.info(f"✓ Tags configured: git_sha={tags.git_sha}, branch={tags.branch}\n")

logger.info(f"✓ Tags configured: git_sha={tags.git_sha}, branch={tags.branch}\n")

# COMMAND ----------
# Initialize Model

logger.info("STEP 1: Model Initialization")
logger.info("-" * 60)

init_start = time.time()

try:
    # Initialize model with configuration
    basic_model = BasicModel(config=config, tags=tags, spark=spark)
    
    logger.info("✓ BasicModel initialized successfully")
    logger.info(f"  Model name: {basic_model.model_name}")
    logger.info(f"  Experiment: {basic_model.experiment_name}")
    logger.info(f"  Parameters: {list(basic_model.parameters.keys())}")
    
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

init_time = time.time() - init_start
logger.info(f"\nInitialization time: {init_time:.2f}s\n")

# COMMAND ----------
# Load and Prepare Data

logger.info("STEP 2: Data Loading and Feature Preparation")
logger.info("-" * 60)

data_start = time.time()

try:
    # Load training and test data
    logger.info("Loading data from Databricks tables...")
    basic_model.load_data()
    
    logger.info("✓ Data loaded successfully")
    logger.info(f"  Train set: {basic_model.X_train.shape}")
    logger.info(f"  Test set: {basic_model.X_test.shape}")
    logger.info(f"  Train data version: {basic_model.train_data_version}")
    logger.info(f"  Test data version: {basic_model.test_data_version}")
    logger.info("  Target distribution (train):")
    for value, count in basic_model.y_train.value_counts().items():
        logger.info(f"    {value}: {count:,} ({count/len(basic_model.y_train)*100:.1f}%)")
    
    # Prepare features
    logger.info("\nPreparing feature pipeline...")
    basic_model.prepare_features()
    
    logger.info("✓ Feature pipeline prepared")
    logger.info(f"  Pipeline steps: {len(basic_model.pipeline.steps)}")
    for step_name, _ in basic_model.pipeline.steps:
        logger.info(f"    - {step_name}")
    
except Exception as e:
    logger.error(f"Failed during data loading/preparation: {e}")
    raise

data_time = time.time() - data_start
logger.info(f"\nData preparation time: {data_time:.2f}s\n")

data_time = time.time() - data_start
logger.info(f"\nData preparation time: {data_time:.2f}s\n")

# COMMAND ----------
# Train Model

logger.info("STEP 3: Model Training")
logger.info("-" * 60)

train_start = time.time()

try:
    logger.info("Starting model training...")
    basic_model.train()
    
    logger.info("✓ Model trained successfully")
    
    # Get model predictions for evaluation
    train_predictions = basic_model.pipeline.predict(basic_model.X_train)
    test_predictions = basic_model.pipeline.predict(basic_model.X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(basic_model.y_train, train_predictions)
    test_accuracy = accuracy_score(basic_model.y_test, test_predictions)
    train_f1 = f1_score(basic_model.y_train, train_predictions)
    test_f1 = f1_score(basic_model.y_test, test_predictions)
    
    logger.info(f"  Training accuracy: {train_accuracy:.4f}")
    logger.info(f"  Training F1-score: {train_f1:.4f}")
    logger.info(f"  Test accuracy: {test_accuracy:.4f}")
    logger.info(f"  Test F1-score: {test_f1:.4f}")
    
except Exception as e:
    logger.error(f"Failed during model training: {e}")
    raise

train_time = time.time() - train_start
logger.info(f"\nTraining time: {train_time:.2f}s\n")

# COMMAND ----------
# Log Model with MLflow

logger.info("STEP 4: Model Logging with MLflow")
logger.info("-" * 60)

log_start = time.time()

try:
    logger.info("Logging model to MLflow...")
    basic_model.log_model()
    
    logger.info("✓ Model logged successfully")
    logger.info(f"  Run ID: {basic_model.run_id}")
    logger.info(f"  Model URI: {basic_model.model_info.model_uri}")
    logger.info(f"  Artifact path: {basic_model.model_info.artifact_path}")
    
    logger.info("\n  MLflow Evaluation Metrics:")
    for metric_name, metric_value in basic_model.metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"    {metric_name}: {metric_value:.4f}")
    
except Exception as e:
    logger.error(f"Failed during model logging: {e}")
    raise

log_time = time.time() - log_start
logger.info(f"\nModel logging time: {log_time:.2f}s\n")

log_time = time.time() - log_start
logger.info(f"\nModel logging time: {log_time:.2f}s\n")

# COMMAND ----------
# Retrieve and Inspect Logged Model

logger.info("STEP 5: Model Inspection and Metadata")
logger.info("-" * 60)

inspect_start = time.time()

try:
    # Retrieve logged model metadata
    logger.info("Retrieving logged model metadata...")
    logged_model = mlflow.get_logged_model(basic_model.model_info.model_id)
    
    logger.info("✓ Model metadata retrieved")
    logger.info(f"  Model ID: {basic_model.model_info.model_id}")
    logger.info(f"  Flavors: {list(logged_model.flavors.keys())}")
    
    # Load the model
    logger.info("\nLoading model for inference...")
    model = mlflow.sklearn.load_model(f"models:/{basic_model.model_info.model_id}")
    logger.info(f"✓ Model loaded: {type(model)}")
    
    # Save model metadata to file
    artifacts_dir = Path("../demo_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    logged_model_dict = logged_model.to_dictionary()
    logged_model_dict["metrics"] = [x.__dict__ for x in logged_model_dict["metrics"]]
    
    model_metadata_path = artifacts_dir / "logged_model.json"
    with open(model_metadata_path, "w", encoding="utf-8") as json_file:
        json.dump(logged_model_dict, json_file, indent=4)
    
    logger.info(f"✓ Model metadata saved to {model_metadata_path}")
    
    # Display parameters and metrics
    logger.info("\n  Model Parameters:")
    for param in logged_model.params[:5]:  # Show first 5
        logger.info(f"    {param.key}: {param.value}")
    if len(logged_model.params) > 5:
        logger.info(f"    ... and {len(logged_model.params) - 5} more")
    
    logger.info("\n  Model Metrics:")
    for metric in logged_model.metrics:
        logger.info(f"    {metric.key}: {metric.value:.4f}")
    
except Exception as e:
    logger.error(f"Failed during model inspection: {e}")
    raise

inspect_time = time.time() - inspect_start
logger.info(f"\nModel inspection time: {inspect_time:.2f}s\n")

# COMMAND ----------
# Retrieve Model by Run ID

logger.info("STEP 6: Model Retrieval by Run ID")
logger.info("-" * 60)

try:
    logger.info(f"Searching for runs in experiment: {config.experiment_name_basic}")
    logger.info(f"Filter: tags.git_sha='{tags.git_sha}'")
    
    # Search for runs with specific git SHA
    search_results = mlflow.search_runs(
        experiment_names=[config.experiment_name_basic],
        filter_string=f"tags.git_sha='{tags.git_sha}'"
    )
    
    if not search_results.empty:
        run_id = search_results.run_id.iloc[0]
        logger.info(f"✓ Found run: {run_id}")
        
        # Load model by run ID
        model_by_run = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")
        logger.info(f"✓ Model loaded from run: {type(model_by_run)}")
        
        # Get run details
        run = mlflow.get_run(run_id)
        logger.info(f"  Run name: {run.info.run_name}")
        logger.info(f"  Status: {run.info.status}")
        logger.info(f"  Start time: {run.info.start_time}")
    else:
        logger.warning("No runs found matching the criteria")
    
except Exception as e:
    logger.error(f"Failed during run retrieval: {e}")
    raise

logger.info("")

# COMMAND ----------
# Inspect Dataset Inputs

logger.info("STEP 7: Dataset Input Inspection")
logger.info("-" * 60)

try:
    # Get dataset inputs from the run
    run = mlflow.get_run(basic_model.run_id)
    inputs = run.inputs.dataset_inputs
    
    logger.info(f"✓ Found {len(inputs)} dataset inputs")
    
    # Retrieve training dataset
    training_input = next((x for x in inputs if x.tags[0].value == 'training'), None)
    if training_input:
        logger.info("\n  Training Dataset:")
        logger.info(f"    Name: {training_input.dataset.name}")
        logger.info(f"    Source: {training_input.dataset.source}")
        training_source = mlflow.data.get_source(training_input)
        training_data = training_source.load()
        logger.info(f"    Shape: {training_data.shape}")
    
    # Retrieve testing dataset
    testing_input = next((x for x in inputs if x.tags[0].value == 'testing'), None)
    if testing_input:
        logger.info("\n  Testing Dataset:")
        logger.info(f"    Name: {testing_input.dataset.name}")
        logger.info(f"    Source: {testing_input.dataset.source}")
        testing_source = mlflow.data.get_source(testing_input)
        testing_data = testing_source.load()
        logger.info(f"    Shape: {testing_data.shape}")
    
except Exception as e:
    logger.error(f"Failed during dataset inspection: {e}")
    raise

logger.info("")

logger.info("")

# COMMAND ----------
# Register Model to Unity Catalog

logger.info("STEP 8: Model Registration")
logger.info("-" * 60)

register_start = time.time()

try:
    logger.info("Registering model to Unity Catalog...")
    logger.info(f"  Model name: {basic_model.model_name}")
    
    model_version = basic_model.register_model()
    
    logger.info("✓ Model registered successfully")
    logger.info(f"  Model name: {basic_model.model_name}")
    logger.info(f"  Version: {model_version}")
    logger.info("  Alias: latest-model")
    
except Exception as e:
    logger.error(f"Failed during model registration: {e}")
    raise

register_time = time.time() - register_start
logger.info(f"\nModel registration time: {register_time:.2f}s\n")

# COMMAND ----------
# Search Model Versions

logger.info("STEP 9: Model Version Search")
logger.info("-" * 60)

try:
    # Search by model name (supported)
    logger.info(f"Searching model versions by name: {basic_model.model_name}")
    versions = mlflow.search_model_versions(
        filter_string=f"name='{basic_model.model_name}'"
    )
    
    if versions:
        logger.info(f"✓ Found {len(versions)} version(s)")
        latest_version = versions[0]
        logger.info("\n  Latest Version Details:")
        logger.info(f"    Version: {latest_version.version}")
        logger.info(f"    Status: {latest_version.status}")
        logger.info(f"    Source: {latest_version.source}")
        logger.info(f"    Run ID: {latest_version.run_id}")
        if hasattr(latest_version, 'creation_timestamp'):
            logger.info(f"    Created: {latest_version.creation_timestamp}")
    else:
        logger.warning("No model versions found")
    
    # Note: Searching by tags is not supported in Unity Catalog
    logger.info("\nNote: Searching model versions by tags (e.g., git_sha) is not supported in Unity Catalog")
    
except Exception as e:
    logger.error(f"Failed during model version search: {e}")
    raise

logger.info("")

# COMMAND ----------
# Pipeline Summary

pipeline_total_time = time.time() - pipeline_start

logger.info("="*60)
logger.info("PIPELINE EXECUTION SUMMARY")
logger.info("="*60)
logger.info("✓ Basic model training and registration completed!")
logger.info("")
logger.info("Execution Times:")
logger.info(f"  Initialization:      {init_time:.2f}s")
logger.info(f"  Data Preparation:    {data_time:.2f}s")
logger.info(f"  Model Training:      {train_time:.2f}s")
logger.info(f"  Model Logging:       {log_time:.2f}s")
logger.info(f"  Model Inspection:    {inspect_time:.2f}s")
logger.info(f"  Model Registration:  {register_time:.2f}s")
logger.info(f"  {'─'*25}")
logger.info(f"  Total Time:          {pipeline_total_time:.2f}s")
logger.info("")
logger.info("Model Performance:")
logger.info(f"  Training F1-score:   {train_f1:.4f}")
logger.info(f"  Test F1-score:       {test_f1:.4f}")
logger.info(f"  Training Accuracy:   {train_accuracy:.4f}")
logger.info(f"  Test Accuracy:       {test_accuracy:.4f}")
logger.info("")
logger.info("Model Details:")
logger.info(f"  Model Name:  {basic_model.model_name}")
logger.info(f"  Version:     {model_version}")
logger.info(f"  Run ID:      {basic_model.run_id}")
logger.info(f"  Experiment:  {basic_model.experiment_name}")
logger.info("")
logger.info(f"Registry Location: {config.catalog_name}.{config.schema_name}")
logger.info("="*60)

# COMMAND ----------
