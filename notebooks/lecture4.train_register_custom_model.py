# Databricks notebook source

"""Custom Model Training and Registration Pipeline.

This notebook demonstrates:
- Loading a pre-trained basic model
- Wrapping it in a custom MLflow PyFunc model
- Registering the custom model to Unity Catalog
- Model inference and unwrapping for predictions
"""

import os
import time
from importlib.metadata import version

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import MarvelModelWrapper


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
logger.info("CUSTOM MODEL TRAINING & REGISTRATION PIPELINE")
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
    logger.info(f"  Experiment: {config.experiment_name_custom}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Initialize Spark session
spark = SparkSession.builder.appName("MarvelCustomModelRegistration").getOrCreate()
logger.info(f"✓ Spark session initialized (version {spark.version})")

# Setup tags for tracking
tags = Tags(git_sha="abcd12345", branch="main")
logger.info(f"✓ Tags configured: git_sha={tags.git_sha}, branch={tags.branch}")

# Get package version and prepare code paths
marvel_characters_v = version("marvel_characters")
code_paths = [f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]
logger.info(f"✓ Package version: {marvel_characters_v}")
logger.info(f"  Code paths: {code_paths}\n")

logger.info(f"✓ Package version: {marvel_characters_v}")
logger.info(f"  Code paths: {code_paths}\n")

# COMMAND ----------
# Load Base Model from Registry

logger.info("STEP 1: Loading Base Model")
logger.info("-" * 60)

load_start = time.time()

try:
    client = MlflowClient()
    
    # Retrieve the latest version of the basic model
    basic_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic"
    logger.info(f"Fetching model: {basic_model_name}")
    
    wrapped_model_version = client.get_model_version_by_alias(
        name=basic_model_name,
        alias="latest-model"
    )
    
    logger.info("✓ Base model loaded successfully")
    logger.info(f"  Model ID: {wrapped_model_version.model_id}")
    logger.info(f"  Version: {wrapped_model_version.version}")
    logger.info(f"  Status: {wrapped_model_version.status}")
    logger.info(f"  Source: {wrapped_model_version.source}")
    
except Exception as e:
    logger.error(f"Failed to load base model: {e}")
    raise

load_time = time.time() - load_start
logger.info(f"\nModel loading time: {load_time:.2f}s\n")

# COMMAND ----------
# Load Test Data

logger.info("STEP 2: Loading Test Data")
logger.info("-" * 60)

data_start = time.time()

try:
    # Load test set from catalog
    test_table_name = f"{config.catalog_name}.{config.schema_name}.test_set"
    logger.info(f"Loading test data from: {test_table_name}")
    
    test_set = spark.table(test_table_name).toPandas()
    X_test = test_set[config.num_features + config.cat_features]
    
    logger.info("✓ Test data loaded successfully")
    logger.info(f"  Test set shape: {test_set.shape}")
    logger.info(f"  Features shape: {X_test.shape}")
    logger.info(f"  Numerical features: {len(config.num_features)}")
    logger.info(f"  Categorical features: {len(config.cat_features)}")
    logger.info(f"  Memory usage: {X_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
except Exception as e:
    logger.error(f"Failed to load test data: {e}")
    raise

data_time = time.time() - data_start
logger.info(f"\nData loading time: {data_time:.2f}s\n")

data_time = time.time() - data_start
logger.info(f"\nData loading time: {data_time:.2f}s\n")

# COMMAND ----------
# Wrap and Register Custom Model

logger.info("STEP 3: Model Wrapping and Registration")
logger.info("-" * 60)

register_start = time.time()

try:
    # Define custom model name
    pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom"
    logger.info(f"Custom model name: {pyfunc_model_name}")
    
    # Initialize wrapper
    wrapper = MarvelModelWrapper()
    logger.info("✓ Model wrapper initialized")
    
    # Prepare input example for signature inference
    input_example = X_test.iloc[0:1]
    logger.info(f"Input example shape: {input_example.shape}")
    
    # Log and register the wrapped model
    logger.info("\nLogging and registering custom model...")
    logger.info(f"  Base model URI: models:/{wrapped_model_version.model_id}")
    logger.info(f"  Experiment: {config.experiment_name_custom}")
    
    model_version = wrapper.log_register_model(
        wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
        pyfunc_model_name=pyfunc_model_name,
        experiment_name=config.experiment_name_custom,
        input_example=input_example,
        tags=tags,
        code_paths=code_paths
    )
    
    logger.info("✓ Custom model registered successfully")
    logger.info(f"  Model name: {pyfunc_model_name}")
    logger.info(f"  Version: {model_version}")
    logger.info("  Alias: latest-model")
    
except Exception as e:
    logger.error(f"Failed to wrap and register model: {e}")
    raise

register_time = time.time() - register_start
logger.info(f"\nModel registration time: {register_time:.2f}s\n")

# COMMAND ----------
# Load and Test Custom Model

logger.info("STEP 4: Model Inference Testing")
logger.info("-" * 60)

inference_start = time.time()

try:
    # Load the registered custom model
    logger.info(f"Loading custom model: {pyfunc_model_name}@latest-model")
    loaded_pyfunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")
    
    logger.info("✓ Custom model loaded successfully")
    logger.info(f"  Model type: {type(loaded_pyfunc_model)}")
    
    # Unwrap the Python model
    unwrapped_model = loaded_pyfunc_model.unwrap_python_model()
    logger.info(f"✓ Model unwrapped: {type(unwrapped_model)}")
    
    # Test prediction with a single sample
    test_sample = X_test.iloc[0:1]
    logger.info(f"\nTesting prediction with sample shape: {test_sample.shape}")
    
    prediction = unwrapped_model.predict(context=None, model_input=test_sample)
    
    logger.info("✓ Prediction successful")
    logger.info(f"  Input features: {list(test_sample.columns)[:5]}... ({len(test_sample.columns)} total)")
    logger.info(f"  Prediction output: {prediction}")
    
    # Test batch prediction
    logger.info(f"\nTesting batch prediction with {len(X_test)} samples...")
    batch_predictions = unwrapped_model.predict(context=None, model_input=X_test)
    
    logger.info("✓ Batch prediction successful")
    if isinstance(batch_predictions, dict) and "Survival prediction" in batch_predictions:
        pred_values = batch_predictions["Survival prediction"]
        alive_count = sum(1 for p in pred_values if p == "alive")
        dead_count = sum(1 for p in pred_values if p == "dead")
        logger.info(f"  Total predictions: {len(pred_values)}")
        logger.info(f"  Alive predictions: {alive_count} ({alive_count/len(pred_values)*100:.1f}%)")
        logger.info(f"  Dead predictions: {dead_count} ({dead_count/len(pred_values)*100:.1f}%)")
    
except Exception as e:
    logger.error(f"Failed during model inference: {e}")
    raise

inference_time = time.time() - inference_start
logger.info(f"\nInference testing time: {inference_time:.2f}s\n")

# COMMAND ----------
# Pipeline Summary

pipeline_total_time = time.time() - pipeline_start

logger.info("="*60)
logger.info("PIPELINE EXECUTION SUMMARY")
logger.info("="*60)
logger.info("✓ Custom model training and registration completed!")
logger.info("")
logger.info("Execution Times:")
logger.info(f"  Base Model Loading:  {load_time:.2f}s")
logger.info(f"  Test Data Loading:   {data_time:.2f}s")
logger.info(f"  Model Registration:  {register_time:.2f}s")
logger.info(f"  Inference Testing:   {inference_time:.2f}s")
logger.info(f"  {'─'*25}")
logger.info(f"  Total Time:          {pipeline_total_time:.2f}s")
logger.info("")
logger.info("Model Details:")
logger.info(f"  Base Model:    {basic_model_name}")
logger.info(f"  Custom Model:  {pyfunc_model_name}")
logger.info(f"  Version:       {model_version}")
logger.info(f"  Package:       marvel_characters v{marvel_characters_v}")
logger.info("")
logger.info(f"Registry Location: {config.catalog_name}.{config.schema_name}")
logger.info("="*60)

# COMMAND ----------