# Databricks notebook source
# MAGIC %pip install marvelousmlops-marvel-characters-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

"""
Marvel Characters A/B Testing Notebook

This notebook demonstrates A/B testing with MLflow by:
1. Training two model variants (A and B) with different hyperparameters
2. Creating a PyFunc wrapper that routes predictions based on hash-based splitting
3. Deploying the A/B testing model to a serving endpoint
4. Testing the endpoint with sample predictions
"""

import hashlib
import os
import time
from typing import Any, Dict, List, Tuple

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
from marvel_characters.utils import is_databricks

# COMMAND ----------

# STEP 1: Initialize Environment and Configuration
logger.info("=" * 80)
logger.info("STEP 1: Initializing Environment and Configuration")
logger.info("=" * 80)

try:
    if not is_databricks():
        logger.info("Running in local environment - loading .env configuration")
        load_dotenv()
        
        # Validate required environment variables
        dbr_token = os.environ.get("DBR_TOKEN")
        dbr_host = os.environ.get("DBR_HOST")
        profile = os.environ.get("PROFILE")
        
        if not dbr_token or not dbr_host:
            raise ValueError("DBR_TOKEN and DBR_HOST must be set in your .env file")
        if not profile:
            raise ValueError("PROFILE must be set in your .env file")
            
        logger.info(f"Setting MLflow tracking URI: databricks://{profile}")
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
    else:
        logger.info("Running in Databricks environment")
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
        logger.info(f"Databricks host: {os.environ['DBR_HOST']}")

    # Load configuration
    config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
    logger.info(f"Loaded configuration for catalog: {config.catalog_name}.{config.schema_name}")
    
    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info(f"Spark session initialized: {spark.version}")
    
    # Define tags for tracking
    tags = Tags(git_sha="dev", branch="ab-testing")
    logger.info(f"Tags configured: {tags}")
    
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    
    logger.info("✓ Environment initialization completed successfully")
    
except Exception as e:
    logger.error(f"✗ Failed to initialize environment: {e}")
    raise

# COMMAND ----------

# STEP 2: Train Model A (Baseline Model)
logger.info("=" * 80)
logger.info("STEP 2: Training Model A (Baseline)")
logger.info("=" * 80)

try:
    start_time = time.time()
    
    logger.info("Initializing BasicModel A with default hyperparameters")
    basic_model_a = BasicModel(config=config, tags=tags, spark=spark)
    
    logger.info("Loading training data...")
    basic_model_a.load_data()
    
    logger.info("Preparing features...")
    basic_model_a.prepare_features()
    
    logger.info("Training Model A...")
    basic_model_a.train()
    
    logger.info("Logging Model A to MLflow...")
    basic_model_a.log_model()
    
    logger.info("Registering Model A to Unity Catalog...")
    basic_model_a.register_model()
    
    model_A_uri = f"models:/{basic_model_a.model_name}@latest-model"
    elapsed = time.time() - start_time
    
    logger.info(f"Model A Name: {basic_model_a.model_name}")
    logger.info(f"Model A URI: {model_A_uri}")
    logger.info(f"✓ Model A training completed in {elapsed:.2f} seconds")
    
except Exception as e:
    logger.error(f"✗ Failed to train Model A: {e}")
    raise

# COMMAND ----------

# STEP 3: Train Model B (Variant Model)
logger.info("=" * 80)
logger.info("STEP 3: Training Model B (Variant)")
logger.info("=" * 80)

try:
    start_time = time.time()
    
    logger.info("Initializing BasicModel B with modified hyperparameters")
    basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
    
    # Configure different hyperparameters for Model B
    basic_model_b.parameters = {
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "max_depth": 6
    }
    basic_model_b.model_name = f"{catalog_name}.{schema_name}.marvel_character_model_basic_B"
    
    logger.info(f"Model B hyperparameters: {basic_model_b.parameters}")
    logger.info(f"Model B name: {basic_model_b.model_name}")
    
    logger.info("Loading training data...")
    basic_model_b.load_data()
    
    logger.info("Preparing features...")
    basic_model_b.prepare_features()
    
    logger.info("Training Model B...")
    basic_model_b.train()
    
    logger.info("Logging Model B to MLflow...")
    basic_model_b.log_model()
    
    logger.info("Registering Model B to Unity Catalog...")
    basic_model_b.register_model()
    
    model_B_uri = f"models:/{basic_model_b.model_name}@latest-model"
    elapsed = time.time() - start_time
    
    logger.info(f"Model B Name: {basic_model_b.model_name}")
    logger.info(f"Model B URI: {model_B_uri}")
    logger.info(f"✓ Model B training completed in {elapsed:.2f} seconds")
    
except Exception as e:
    logger.error(f"✗ Failed to train Model B: {e}")
    raise

# COMMAND ----------

# STEP 4: Define A/B Test Wrapper
logger.info("=" * 80)
logger.info("STEP 4: Defining A/B Test Wrapper")
logger.info("=" * 80)


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A/B Testing PyFunc wrapper that routes predictions to Model A or B.
    
    Routing logic:
    - Uses MD5 hash of the Id field for deterministic splitting
    - Odd hash values → Model A
    - Even hash values → Model B
    """
    
    def load_context(self, context):
        """Load both models from the artifacts."""
        logger.info("Loading Model A and Model B from artifacts")
        self.model_a = mlflow.sklearn.load_model(
            context.artifacts["sklearn-pipeline-model-A"]
        )
        self.model_b = mlflow.sklearn.load_model(
            context.artifacts["sklearn-pipeline-model-B"]
        )
        logger.info("Both models loaded successfully")

    def predict(self, context, model_input):  # noqa: ARG002
        """
        Route prediction to Model A or B based on hash of Id field.
        
        Args:
            context: MLflow context (unused)
            model_input: DataFrame with Id column for routing
            
        Returns:
            Dictionary with prediction and model identifier
        """
        # Use PageID (or another unique identifier) for splitting
        page_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(page_id.encode(encoding="UTF-8")).hexdigest()
        
        if int(hashed_id, 16) % 2:
            # Odd hash → Model A
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            # Even hash → Model B
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}


logger.info("✓ MarvelModelWrapper class defined successfully")
logger.info("Routing logic: Hash-based splitting on Id field (odd=A, even=B)")

# COMMAND ----------

# STEP 5: Prepare Data for Model Signature
logger.info("=" * 80)
logger.info("STEP 5: Preparing Data for Model Signature")
logger.info("=" * 80)

try:
    logger.info(f"Loading train_set from {catalog_name}.{schema_name}.train_set")
    train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
    train_set = train_set_spark.toPandas()
    
    logger.info(f"Loading test_set from {catalog_name}.{schema_name}.test_set")
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
    
    # Select features with Id for routing
    feature_cols = config.num_features + config.cat_features + ["Id"]
    X_train = train_set[feature_cols]
    X_test = test_set[feature_cols]
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    logger.info("✓ Data preparation completed successfully")
    
except Exception as e:
    logger.error(f"✗ Failed to prepare data: {e}")
    raise

# COMMAND ----------

# STEP 6: Log and Register A/B Testing Model
logger.info("=" * 80)
logger.info("STEP 6: Logging and Registering A/B Testing Model")
logger.info("=" * 80)

try:
    start_time = time.time()
    
    experiment_name = "/Shared/marvel-characters-ab-testing"
    logger.info(f"Setting MLflow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name=experiment_name)
    
    model_name = f"{catalog_name}.{schema_name}.marvel_character_model_pyfunc_ab_test"
    logger.info(f"Model name: {model_name}")
    
    wrapped_model = MarvelModelWrapper()
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        # Infer signature with sample output
        logger.info("Inferring model signature...")
        signature = infer_signature(
            model_input=X_train, 
            model_output={"Prediction": 1, "model": "Model B"}
        )
        
        # Log dataset lineage
        logger.info("Logging input dataset...")
        dataset = mlflow.data.from_spark(
            train_set_spark, 
            table_name=f"{catalog_name}.{schema_name}.train_set", 
            version="0"
        )
        mlflow.log_input(dataset, context="training")
        
        # Log the PyFunc model with both model artifacts
        logger.info("Logging PyFunc model with Model A and Model B artifacts...")
        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path="pyfunc-marvel-character-model-ab",
            artifacts={
                "sklearn-pipeline-model-A": model_A_uri,
                "sklearn-pipeline-model-B": model_B_uri
            },
            signature=signature
        )
        logger.info("PyFunc model logged successfully")
    
    # Register model to Unity Catalog
    logger.info(f"Registering model to Unity Catalog: {model_name}")
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/pyfunc-marvel-character-model-ab", 
        name=model_name
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Model registered with version: {model_version.version}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"✓ A/B testing model logged and registered in {elapsed:.2f} seconds")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to log and register A/B testing model: {e}")
    raise

# COMMAND ----------

# STEP 7: Deploy Model to Serving Endpoint
logger.info("=" * 80)
logger.info("STEP 7: Deploying Model to Serving Endpoint")
logger.info("=" * 80)

try:
    start_time = time.time()
    
    workspace = WorkspaceClient()
    endpoint_name = "marvel-characters-ab-testing"
    entity_version = model_version.version
    
    logger.info(f"Endpoint name: {endpoint_name}")
    logger.info(f"Model version to deploy: {entity_version}")
    
    served_entities = [
        ServedEntityInput(
            entity_name=model_name,
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=entity_version,
        )
    ]
    
    logger.info("Creating serving endpoint...")
    logger.info("Configuration: scale_to_zero=True, workload_size=Small")
    
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
    )
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Serving endpoint '{endpoint_name}' created successfully in {elapsed:.2f} seconds")
    logger.info("Note: Endpoint may take several minutes to become ready")
    
except Exception as e:
    logger.error(f"✗ Failed to create serving endpoint: {e}")
    logger.warning("Endpoint may already exist or there may be a deployment issue")
    # Don't raise - endpoint might already exist

# COMMAND ----------

# STEP 8: Prepare Sample Requests
logger.info("=" * 80)
logger.info("STEP 8: Preparing Sample Requests")
logger.info("=" * 80)

try:
    logger.info("Creating sample records for testing...")
    feature_cols = config.num_features + config.cat_features + ["Id"]
    sampled_records = train_set[feature_cols].sample(n=1000, replace=True).to_dict(orient="records")
    dataframe_records = [[record] for record in sampled_records]
    
    logger.info(f"Generated {len(dataframe_records)} sample records")
    logger.info(f"Sample record structure: {dataframe_records[0]}")
    logger.info(f"Train set dtypes:\n{train_set.dtypes}")
    logger.info("✓ Sample requests prepared successfully")
    
except Exception as e:
    logger.error(f"✗ Failed to prepare sample requests: {e}")
    raise

# COMMAND ----------

# STEP 9: Test Endpoint with Single Request
logger.info("=" * 80)
logger.info("STEP 9: Testing Endpoint with Single Request")
logger.info("=" * 80)


def call_endpoint(endpoint_record: List[Dict[str, Any]]) -> Tuple[int, str]:
    """
    Call the model serving endpoint with a given input record.
    
    Args:
        endpoint_record: List containing a single record dictionary
        
    Returns:
        Tuple of (status_code, response_text)
    """
    serving_endpoint = (
        f"https://{os.environ['DBR_HOST']}/serving-endpoints/"
        f"marvel-characters-ab-testing/invocations"
    )

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": endpoint_record},
    )
    return response.status_code, response.text


try:
    logger.info("Sending test request to endpoint...")
    status_code, response_text = call_endpoint(dataframe_records[0])
    
    logger.info(f"Response Status: {status_code}")
    logger.info(f"Response Text: {response_text}")
    
    if status_code == 200:
        logger.info("✓ Single request test successful")
    else:
        logger.warning(f"Unexpected status code: {status_code}")
        
except Exception as e:
    logger.error(f"✗ Failed to test endpoint: {e}")
    raise

# COMMAND ----------

# STEP 10: Load Test with Multiple Requests
logger.info("=" * 80)
logger.info("STEP 10: Running Load Test with Multiple Requests")
logger.info("=" * 80)

try:
    logger.info(f"Sending {len(dataframe_records)} requests to endpoint...")
    
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    model_a_count = 0
    model_b_count = 0
    
    for i, record in enumerate(dataframe_records):
        try:
            status_code, response_text = call_endpoint(record)
            
            if status_code == 200:
                successful_requests += 1
                # Count which model was used
                if '"model": "Model A"' in response_text or "'model': 'Model A'" in response_text:
                    model_a_count += 1
                elif '"model": "Model B"' in response_text or "'model': 'Model B'" in response_text:
                    model_b_count += 1
            else:
                failed_requests += 1
                logger.warning(f"Request {i+1} failed with status {status_code}: {response_text}")
            
            # Log progress every 100 requests
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(dataframe_records)} requests sent")
            
            # Rate limiting
            time.sleep(0.2)
            
        except Exception as e:  # noqa: BLE001
            failed_requests += 1
            logger.error(f"Request {i+1} raised exception: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary statistics
    logger.info("=" * 80)
    logger.info("Load Test Results Summary")
    logger.info("=" * 80)
    logger.info(f"Total requests: {len(dataframe_records)}")
    logger.info(f"Successful: {successful_requests}")
    logger.info(f"Failed: {failed_requests}")
    logger.info(f"Success rate: {successful_requests / len(dataframe_records) * 100:.2f}%")
    logger.info(f"Model A predictions: {model_a_count}")
    logger.info(f"Model B predictions: {model_b_count}")
    logger.info(f"A/B split ratio: {model_a_count}:{model_b_count}")
    logger.info(f"Total elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Average latency: {elapsed / len(dataframe_records) * 1000:.2f} ms per request")
    logger.info("✓ Load test completed successfully")
    
except Exception as e:
    logger.error(f"✗ Load test failed: {e}")
    raise

# COMMAND ----------

logger.info("=" * 80)
logger.info("A/B Testing Notebook Completed Successfully")
logger.info("=" * 80)
logger.info("Summary:")
logger.info(f"  - Model A: {basic_model_a.model_name}")
logger.info(f"  - Model B: {basic_model_b.model_name}")
logger.info(f"  - A/B Test Model: {model_name}")
logger.info(f"  - Serving Endpoint: {endpoint_name}")
logger.info("=" * 80)
