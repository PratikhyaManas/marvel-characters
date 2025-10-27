"""
Train and Register Custom Model Script

This script trains the Marvel Characters BasicModel and registers it to Unity Catalog
if the model shows improvement over the current production model. It tracks the model
with MLflow and returns the model version for deployment.

Usage:
    python train_register_custom_model.py --root_path <path> --env <environment> \\
        --git_sha <sha> --job_run_id <id> --branch <branch>

Arguments:
    --root_path: Root path to the project files
    --env: Environment name (dev, staging, prod)
    --git_sha: Git SHA of the commit
    --job_run_id: Databricks job run ID
    --branch: Git branch name
"""

import argparse
import sys
from typing import Optional

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train and register Marvel Characters custom model"
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
    parser.add_argument(
        "--git_sha",
        type=str,
        required=True,
        help="Git SHA of the commit"
    )
    parser.add_argument(
        "--job_run_id",
        type=str,
        required=True,
        help="Databricks job run ID"
    )
    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        help="Git branch name"
    )
    
    return parser.parse_args()


def set_task_values(dbutils: DBUtils, model_version: Optional[str], model_updated: bool) -> None:
    """
    Set task values for downstream Databricks jobs.
    
    Args:
        dbutils: Databricks utilities instance
        model_version: Version of the registered model (if updated)
        model_updated: Whether the model was updated
    """
    try:
        if model_updated and model_version:
            dbutils.jobs.taskValues.set(key="model_version", value=model_version)
            dbutils.jobs.taskValues.set(key="model_updated", value=1)
            logger.info(f"Task values set: model_version={model_version}, model_updated=1")
        else:
            dbutils.jobs.taskValues.set(key="model_updated", value=0)
            logger.info("Task values set: model_updated=0")
    except Exception as e:
        logger.warning(f"Failed to set task values: {e}")


def train_and_register_model(
    config_path: str,
    env: str,
    git_sha: str,
    job_run_id: str,
    branch: str
) -> tuple[bool, Optional[str]]:
    """
    Train and optionally register the Marvel Characters model.
    
    Args:
        config_path: Path to the project configuration YAML file
        env: Environment name (dev, staging, prod)
        git_sha: Git SHA of the commit
        job_run_id: Databricks job run ID
        branch: Git branch name
        
    Returns:
        Tuple of (model_updated, model_version)
    """
    logger.info("=" * 80)
    logger.info("Marvel Characters Model Training and Registration")
    logger.info("=" * 80)
    
    try:
        # Load project configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = ProjectConfig.from_yaml(config_path=config_path, env=env)
        logger.info(f"✓ Configuration loaded for environment: {env}")
        
        # Initialize Spark and DBUtils
        logger.info("\nInitializing Spark session...")
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        logger.info(f"✓ Spark session initialized: {spark.version}")
        
        # Create tags for tracking
        tags_dict = {
            "git_sha": git_sha,
            "branch": branch,
            "job_run_id": job_run_id
        }
        tags = Tags(**tags_dict)
        logger.info(f"Tags: {tags}")
        
        # Initialize BasicModel
        logger.info("\nInitializing Marvel BasicModel...")
        marvel_model = BasicModel(config=config, tags=tags, spark=spark)
        logger.info("✓ Marvel BasicModel initialized")
        
        # Load data
        logger.info("\nLoading Marvel characters data...")
        marvel_model.load_data()
        logger.info("✓ Marvel data loaded successfully")
        
        # Feature engineering
        logger.info("\nPerforming feature engineering...")
        marvel_model.feature_engineering()
        logger.info("✓ Feature engineering completed")
        
        # Train the model
        logger.info("\nTraining Marvel model...")
        marvel_model.train()
        logger.info("✓ Model training completed")
        
        # Evaluate model improvement
        logger.info("\nEvaluating model performance...")
        model_improved = marvel_model.model_improved()
        logger.info(f"Model evaluation completed: improved={model_improved}")
        
        # Register model if improved
        model_version = None
        if model_improved:
            logger.info("\n Model shows improvement - registering to Unity Catalog...")
            latest_version = marvel_model.register_model()
            model_version = latest_version
            logger.info(f"✓ New model registered with version: {latest_version}")
            
            # Set task values for downstream jobs
            set_task_values(dbutils, model_version, True)
        else:
            logger.info("\n✗ Model did not show improvement - skipping registration")
            set_task_values(dbutils, None, False)
        
        logger.info("=" * 80)
        logger.info("Training and Registration Completed Successfully")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info(f"  - Environment: {env}")
        logger.info(f"  - Model improved: {model_improved}")
        logger.info(f"  - Model registered: {model_improved}")
        logger.info(f"  - Model version: {model_version if model_version else 'N/A'}")
        logger.info(f"  - Git SHA: {git_sha}")
        logger.info(f"  - Branch: {branch}")
        logger.info("=" * 80)
        
        return model_improved, model_version
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Training and Registration Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


def main() -> None:
    """Main execution function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        logger.info(f"Starting training with environment={args.env}, branch={args.branch}")
        
        # Build configuration path
        config_path = f"{args.root_path}/files/project_config_marvel.yml"
        
        # Train and register the model
        model_improved, model_version = train_and_register_model(
            config_path=config_path,
            env=args.env,
            git_sha=args.git_sha,
            job_run_id=args.job_run_id,
            branch=args.branch
        )
        
        if model_improved:
            logger.info(f"✓ Training completed successfully - Model version: {model_version}")
            sys.exit(0)
        else:
            logger.info("Training completed - No model update required")
            sys.exit(0)
        
    except Exception as e:
        logger.exception(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
