"""
Deploy Model Script

This script deploys a trained Marvel Characters model to a Databricks serving endpoint.
It retrieves the model version from the previous training task and creates or updates
the serving endpoint with the specified model version.

Usage:
    python deploy_model.py --root_path <path> --env <environment>

Arguments:
    --root_path: Root path to the project files
    --env: Environment name (dev, staging, prod)
"""

import argparse
import sys
from typing import Optional

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.serving.model_serving import ModelServing


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Deploy Marvel Characters model to serving endpoint"
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


def get_model_version_from_task(dbutils: DBUtils, task_key: str = "train_model") -> Optional[str]:
    """
    Retrieve model version from a previous Databricks task.
    
    Args:
        dbutils: Databricks utilities instance
        task_key: Name of the task that produced the model version
        
    Returns:
        Model version string or None if not found
    """
    try:
        model_version = dbutils.jobs.taskValues.get(taskKey=task_key, key="model_version")
        logger.info(f"Retrieved model version from task '{task_key}': {model_version}")
        return model_version
    except Exception as e:
        logger.error(f"Failed to retrieve model version from task '{task_key}': {e}")
        raise


def deploy_model(
    config_path: str,
    env: str,
    model_version: Optional[str] = None
) -> None:
    """
    Deploy the Marvel Characters model to a serving endpoint.
    
    Args:
        config_path: Path to the project configuration YAML file
        env: Environment name (dev, staging, prod)
        model_version: Optional model version to deploy (uses latest if None)
    """
    logger.info("=" * 80)
    logger.info("Marvel Characters Model Deployment")
    logger.info("=" * 80)
    
    try:
        # Load project configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = ProjectConfig.from_yaml(config_path=config_path, env=env)
        logger.info(f"✓ Configuration loaded for environment: {env}")
        
        catalog_name = config.catalog_name
        schema_name = config.schema_name
        model_name = f"{catalog_name}.{schema_name}.marvel_character_model_basic"
        endpoint_name = f"marvel-characters-model-serving-{env}"
        
        logger.info(f"Model name: {model_name}")
        logger.info(f"Endpoint name: {endpoint_name}")
        logger.info(f"Model version: {model_version if model_version else 'latest'}")
        
        # Initialize Model Serving Manager
        logger.info("Initializing Model Serving Manager...")
        model_serving = ModelServing(
            model_name=model_name,
            endpoint_name=endpoint_name
        )
        logger.info("✓ Model Serving Manager initialized")
        
        # Deploy or update the serving endpoint
        logger.info("Deploying/updating serving endpoint...")
        model_serving.deploy_or_update_serving_endpoint(version=model_version)
        logger.info("✓ Serving endpoint deployment/update initiated successfully")
        
        logger.info("=" * 80)
        logger.info("Deployment completed successfully")
        logger.info("=" * 80)
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Version: {model_version if model_version else 'latest'}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Deployment Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


def main() -> None:
    """Main execution function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        logger.info(f"Starting deployment with root_path={args.root_path}, env={args.env}")
        
        # Build configuration path
        config_path = f"{args.root_path}/files/project_config_marvel.yml"
        
        # Initialize Spark and DBUtils
        logger.info("Initializing Spark session...")
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        logger.info("✓ Spark session initialized")
        
        # Retrieve model version from previous task
        model_version = get_model_version_from_task(dbutils, task_key="train_model")
        
        # Deploy the model
        deploy_model(
            config_path=config_path,
            env=args.env,
            model_version=model_version
        )
        
        sys.exit(0)
        
    except Exception as e:
        logger.exception(f"Deployment script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
