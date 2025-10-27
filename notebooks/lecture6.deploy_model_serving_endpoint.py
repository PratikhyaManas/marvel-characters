# Databricks notebook source
# MAGIC %pip install marvel_characters-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

"""
Marvel Characters Model Serving Deployment Notebook

This notebook demonstrates deploying a model to a Databricks serving endpoint:
1. Initialize environment and configuration
2. Deploy or update the serving endpoint
3. Test the endpoint with sample predictions
4. Run load test with multiple requests
5. Analyze endpoint performance metrics
"""

import os
import time
from typing import Any, Dict, List, Tuple

import mlflow
import requests
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.serving.model_serving import ModelServing
from marvel_characters.utils import is_databricks

# COMMAND ----------

# STEP 1: Initialize Environment and Configuration
logger.info("=" * 80)
logger.info("STEP 1: Initializing Environment and Configuration")
logger.info("=" * 80)

try:
    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info(f"Spark session initialized: {spark.version}")
    
    if is_databricks():
        logger.info("Running in Databricks environment")
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
        logger.info(f"Databricks host: {os.environ['DBR_HOST']}")
    else:
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

    # Load project configuration
    config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    
    logger.info(f"Loaded configuration for catalog: {catalog_name}.{schema_name}")
    logger.info(f"Model name: {catalog_name}.{schema_name}.marvel_character_model_custom")
    logger.info("✓ Environment initialization completed successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to initialize environment: {e}")
    raise

# COMMAND ----------

# STEP 2: Initialize Model Serving
logger.info("=" * 80)
logger.info("STEP 2: Initializing Model Serving")
logger.info("=" * 80)

try:
    model_name = f"{catalog_name}.{schema_name}.marvel_character_model_custom"
    endpoint_name = "marvel-character-model-serving"
    
    logger.info(f"Model name: {model_name}")
    logger.info(f"Endpoint name: {endpoint_name}")
    
    model_serving = ModelServing(
        model_name=model_name,
        endpoint_name=endpoint_name
    )
    
    logger.info("✓ ModelServing instance created successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to initialize model serving: {e}")
    raise

# COMMAND ----------

# STEP 3: Deploy or Update Serving Endpoint
logger.info("=" * 80)
logger.info("STEP 3: Deploying or Updating Serving Endpoint")
logger.info("=" * 80)

try:
    start_time = time.time()
    
    logger.info(f"Deploying endpoint '{endpoint_name}'...")
    logger.info("This may take several minutes for initial deployment")
    
    model_serving.deploy_or_update_serving_endpoint()
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Serving endpoint deployed/updated successfully in {elapsed:.2f} seconds")
    logger.info("Note: Endpoint may need additional time to reach 'Ready' state")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to deploy/update serving endpoint: {e}")
    logger.warning("Endpoint may already exist or there may be a configuration issue")
    raise

# COMMAND ----------

# STEP 4: Prepare Test Data
logger.info("=" * 80)
logger.info("STEP 4: Preparing Test Data")
logger.info("=" * 80)

try:
    # Define required columns for prediction
    required_columns = [
        "Height",
        "Weight",
        "Universe",
        "Identity",
        "Gender",
        "Marital_Status",
        "Teams",
        "Origin",
        "Creators",
    ]
    
    logger.info(f"Required columns: {required_columns}")
    logger.info(f"Loading test set from {config.catalog_name}.{config.schema_name}.test_set")
    
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
    logger.info(f"Test set loaded: {test_set.shape}")
    
    # Sample records for testing (using large sample for load test)
    sample_size = 18000
    logger.info(f"Sampling {sample_size} records (with replacement) for load testing")
    
    sampled_records = test_set[required_columns].sample(n=sample_size, replace=True).to_dict(orient="records")
    dataframe_records = [[record] for record in sampled_records]
    
    logger.info(f"Generated {len(dataframe_records)} test records")
    logger.info(f"Sample record structure:\n{dataframe_records[0]}")
    logger.info("✓ Test data prepared successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to prepare test data: {e}")
    raise

# COMMAND ----------

# STEP 5: Define Endpoint Invocation Function
logger.info("=" * 80)
logger.info("STEP 5: Defining Endpoint Invocation Function")
logger.info("=" * 80)


def call_endpoint(endpoint_record: List[Dict[str, Any]]) -> Tuple[int, str]:
    """
    Call the model serving endpoint with a given input record.
    
    Args:
        endpoint_record: List containing a single record dictionary with feature values
        
    Returns:
        Tuple of (status_code, response_text)
        
    Example record format:
        [{'Height': 1.75,
          'Weight': 70.0,
          'Universe': 'Earth-616',
          'Identity': 'Public',
          'Gender': 'Male',
          'Marital_Status': 'Single',
          'Teams': 'Avengers',
          'Origin': 'Human',
          'Creators': 'Stan Lee'}]
    """
    host = os.environ['DBR_HOST']
    
    # Validate and normalize host URL
    if '.' not in host:
        logger.warning(f"DBR_HOST '{host}' may be incomplete. Adding '.com' domain suffix")
        host = f"{host}.com"
    
    serving_endpoint = (
        f"https://{host}/serving-endpoints/"
        f"marvel-character-model-serving/invocations"
    )
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": endpoint_record},
        timeout=30
    )
    return response.status_code, response.text


logger.info("✓ Endpoint invocation function defined")
logger.info(f"Endpoint URL: https://{os.environ['DBR_HOST']}/serving-endpoints/marvel-character-model-serving/invocations")

# COMMAND ----------

# STEP 6: Test Endpoint with Single Request
logger.info("=" * 80)
logger.info("STEP 6: Testing Endpoint with Single Request")
logger.info("=" * 80)

try:
    logger.info("Sending test request to endpoint...")
    logger.info(f"Test record: {dataframe_records[0]}")
    
    status_code, response_text = call_endpoint(dataframe_records[0])
    
    logger.info(f"Response Status: {status_code}")
    logger.info(f"Response Text: {response_text}")
    
    if status_code == 200:
        logger.info("✓ Single request test successful")
    else:
        logger.warning(f"Unexpected status code: {status_code}")
        logger.warning("Endpoint may still be initializing or there may be an issue")
        
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to test endpoint: {e}")
    raise

# COMMAND ----------

# STEP 7: Run Load Test with Multiple Requests
logger.info("=" * 80)
logger.info("STEP 7: Running Load Test with Multiple Requests")
logger.info("=" * 80)

try:
    logger.info(f"Starting load test with {len(dataframe_records)} requests...")
    logger.info("This will take approximately 1 hour (0.2s delay per request)")
    
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    response_times = []
    predictions = []
    
    for i, record in enumerate(dataframe_records):
        try:
            request_start = time.time()
            status_code, response_text = call_endpoint(record)
            request_elapsed = time.time() - request_start
            
            response_times.append(request_elapsed)
            
            if status_code == 200:
                successful_requests += 1
                # Extract prediction if available
                try:
                    import json  # noqa: PLC0415
                    response_data = json.loads(response_text)
                    if 'predictions' in response_data:
                        predictions.append(response_data['predictions'])
                except Exception:  # noqa: BLE001, S110
                    pass
            else:
                failed_requests += 1
                logger.warning(f"Request {i+1} failed with status {status_code}")
                if failed_requests <= 5:  # Only log first 5 failures
                    logger.warning(f"Response: {response_text}")
            
            # Progress logging every 1000 requests
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                avg_latency = sum(response_times) / len(response_times) * 1000
                logger.info(
                    f"Progress: {i+1}/{len(dataframe_records)} | "
                    f"Success: {successful_requests} | "
                    f"Failed: {failed_requests} | "
                    f"Avg Latency: {avg_latency:.2f}ms | "
                    f"Elapsed: {elapsed:.2f}s"
                )
            
            # Rate limiting to avoid overwhelming the endpoint
            time.sleep(0.2)
            
        except Exception as e:  # noqa: BLE001
            failed_requests += 1
            if failed_requests <= 5:  # Only log first 5 exceptions
                logger.error(f"Request {i+1} raised exception: {e}")
    
    total_elapsed = time.time() - start_time
    
    # Calculate statistics
    avg_latency = sum(response_times) / len(response_times) * 1000 if response_times else 0
    min_latency = min(response_times) * 1000 if response_times else 0
    max_latency = max(response_times) * 1000 if response_times else 0
    
    # Summary statistics
    logger.info("=" * 80)
    logger.info("Load Test Results Summary")
    logger.info("=" * 80)
    logger.info(f"Total requests: {len(dataframe_records)}")
    logger.info(f"Successful: {successful_requests}")
    logger.info(f"Failed: {failed_requests}")
    logger.info(f"Success rate: {successful_requests / len(dataframe_records) * 100:.2f}%")
    logger.info(f"Total elapsed time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    logger.info(f"Average latency: {avg_latency:.2f} ms")
    logger.info(f"Min latency: {min_latency:.2f} ms")
    logger.info(f"Max latency: {max_latency:.2f} ms")
    logger.info(f"Throughput: {len(dataframe_records) / total_elapsed:.2f} requests/second")
    logger.info(f"Total predictions collected: {len(predictions)}")
    logger.info("✓ Load test completed successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Load test failed: {e}")
    raise

# COMMAND ----------

logger.info("=" * 80)
logger.info("Model Serving Deployment Notebook Completed Successfully")
logger.info("=" * 80)
logger.info("Summary:")
logger.info(f"  - Model: {model_name}")
logger.info(f"  - Endpoint: {endpoint_name}")
logger.info("  - Status: Deployed and tested")
logger.info(f"  - Test requests: {len(dataframe_records)}")
logger.info(f"  - Success rate: {successful_requests / len(dataframe_records) * 100:.2f}%")
logger.info("=" * 80)
