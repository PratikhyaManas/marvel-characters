# Databricks notebook source
# MAGIC %pip install marvel_characters-0.1.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Marvel Characters Model Monitoring Setup
# MAGIC 
# MAGIC This notebook demonstrates model monitoring setup:
# MAGIC 1. Send inference requests to the serving endpoint
# MAGIC 2. Create inference logs for monitoring
# MAGIC 3. Set up Lakehouse Monitoring for model performance tracking
# MAGIC 4. Enable data quality and drift detection

# COMMAND ----------

"""
Marvel Characters Model Monitoring Notebook

This notebook sets up monitoring for the deployed model by:
1. Loading test data and sending requests to the serving endpoint
2. Creating inference logs with predictions
3. Setting up Lakehouse Monitoring tables
4. Configuring drift detection and data quality metrics
"""

import os
import time
from typing import List, Tuple

import requests
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from marvel_characters.config import ProjectConfig
from marvel_characters.utils import is_databricks

# COMMAND ----------

# STEP 1: Initialize Environment and Load Data
logger.info("=" * 80)
logger.info("STEP 1: Initializing Environment and Loading Data")
logger.info("=" * 80)

try:
    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info(f"Spark session initialized: {spark.version}")
    
    # Load configuration
    config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
    logger.info(f"Loaded configuration for catalog: {config.catalog_name}.{config.schema_name}")
    
    # Load test set with Id as string for tracking
    logger.info(f"Loading test set from {config.catalog_name}.{config.schema_name}.test_set")
    test_set = (
        spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
        .withColumn("Id", col("Id").cast("string"))
        .toPandas()
    )
    
    logger.info(f"Test set loaded: {test_set.shape}")
    logger.info(f"Columns: {list(test_set.columns)}")
    logger.info("✓ Data loading completed successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to initialize environment and load data: {e}")
    raise

# COMMAND ----------

# STEP 2: Configure Databricks Authentication
logger.info("=" * 80)
logger.info("STEP 2: Configuring Databricks Authentication")
logger.info("=" * 80)

try:
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
        
        logger.info(f"Profile: {profile}")
    
    logger.info("✓ Authentication configured successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to configure authentication: {e}")
    raise

# COMMAND ----------

# STEP 3: Initialize Workspace Client
logger.info("=" * 80)
logger.info("STEP 3: Initializing Workspace Client")
logger.info("=" * 80)

try:
    workspace = WorkspaceClient()
    logger.info("✓ Workspace client initialized successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to initialize workspace client: {e}")
    raise

# COMMAND ----------

# STEP 4: Define Inference Request Functions
logger.info("=" * 80)
logger.info("STEP 4: Defining Inference Request Functions")
logger.info("=" * 80)

# Required columns for inference
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

logger.info(f"Required columns for inference: {required_columns}")


def send_request_https(dataframe_record: List[List[dict]]) -> Tuple[int, str]:
    """
    Send a request to the model serving endpoint using HTTPS.
    
    Args:
        dataframe_record: List containing a single record dictionary
        
    Returns:
        Tuple of (status_code, response_text)
    """
    serving_endpoint = (
        f"https://{os.environ['DBR_HOST']}/serving-endpoints/"
        f"marvel-characters-model-serving/invocations"
    )
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": dataframe_record},
        timeout=30
    )
    return response.status_code, response.text


def send_request_workspace(dataframe_record: List[List[dict]]):
    """
    Send a request to the model serving endpoint using workspace client.
    
    Args:
        dataframe_record: List containing a single record dictionary
        
    Returns:
        Response from the serving endpoint
    """
    response = workspace.serving_endpoints.query(
        name="marvel-characters-model-serving",
        dataframe_records=dataframe_record
    )
    return response


logger.info("✓ Request functions defined successfully")
logger.info("Available methods: send_request_https, send_request_workspace")

# COMMAND ----------

# STEP 5: Prepare Sample Requests
logger.info("=" * 80)
logger.info("STEP 5: Preparing Sample Requests")
logger.info("=" * 80)

try:
    # Sample records for testing
    sample_size = 100
    logger.info(f"Sampling {sample_size} records from test set")
    
    sampled_records = test_set[required_columns].sample(n=sample_size, replace=True).to_dict(orient="records")
    dataframe_records = [[record] for record in sampled_records]
    
    logger.info(f"Generated {len(dataframe_records)} sample records for inference")
    logger.info(f"Sample record: {dataframe_records[0]}")
    logger.info("✓ Sample requests prepared successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to prepare sample requests: {e}")
    raise

# COMMAND ----------

# STEP 6: Send Inference Requests to Endpoint
logger.info("=" * 80)
logger.info("STEP 6: Sending Inference Requests to Endpoint")
logger.info("=" * 80)

try:
    logger.info(f"Sending {len(dataframe_records)} inference requests...")
    logger.info("This will generate inference logs for monitoring")
    
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    
    for i, record in enumerate(dataframe_records):
        try:
            status_code, response_text = send_request_https(record)
            
            if status_code == 200:
                successful_requests += 1
            else:
                failed_requests += 1
                if failed_requests <= 3:  # Only log first 3 failures
                    logger.warning(f"Request {i+1} failed with status {status_code}: {response_text}")
            
            # Progress logging every 20 requests
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(dataframe_records)} requests sent")
            
            # Rate limiting
            time.sleep(0.2)
            
        except Exception as e:  # noqa: BLE001
            failed_requests += 1
            if failed_requests <= 3:
                logger.error(f"Request {i+1} raised exception: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("Inference Request Summary")
    logger.info("=" * 80)
    logger.info(f"Total requests: {len(dataframe_records)}")
    logger.info(f"Successful: {successful_requests}")
    logger.info(f"Failed: {failed_requests}")
    logger.info(f"Success rate: {successful_requests / len(dataframe_records) * 100:.2f}%")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    logger.info("✓ Inference requests completed successfully")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to send inference requests: {e}")
    raise

# COMMAND ----------

# STEP 7: Set Up Lakehouse Monitoring
logger.info("=" * 80)
logger.info("STEP 7: Setting Up Lakehouse Monitoring")
logger.info("=" * 80)

try:
    # Re-import in case of session issues
    from databricks.connect import DatabricksSession  # noqa: PLC0415
    
    from marvel_characters.monitoring import create_or_refresh_monitoring
    
    logger.info("Initializing Databricks session for monitoring setup")
    spark = DatabricksSession.builder.getOrCreate()
    
    # Workspace client already initialized in Step 3
    logger.info(f"Configuration loaded: {config.catalog_name}.{config.schema_name}")
    
    logger.info("Creating or refreshing monitoring tables...")
    logger.info("This will set up:")
    logger.info("  - Inference table for predictions")
    logger.info("  - Profile metrics table")
    logger.info("  - Drift metrics table")
    logger.info("  - Data quality monitoring")
    
    create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
    
    logger.info("✓ Lakehouse Monitoring setup completed successfully")
    logger.info(f"Monitor dashboard available in Catalog: {config.catalog_name}.{config.schema_name}")
    
except Exception as e:  # noqa: BLE001
    logger.error(f"✗ Failed to set up monitoring: {e}")
    raise

# COMMAND ----------

logger.info("=" * 80)
logger.info("Model Monitoring Setup Completed Successfully")
logger.info("=" * 80)
logger.info("Summary:")
logger.info(f"  - Inference requests sent: {len(dataframe_records)}")
logger.info(f"  - Success rate: {successful_requests / len(dataframe_records) * 100:.2f}%")
logger.info("  - Monitoring tables created")
logger.info("  - Drift detection enabled")
logger.info("")
logger.info("Next Steps:")
logger.info("  1. Review the monitoring dashboard in Databricks")
logger.info("  2. Set up alerts for drift or data quality issues")
logger.info("  3. Schedule regular monitoring refreshes")
logger.info("=" * 80)
