"""
Refresh Monitoring Script

This script creates or refreshes Lakehouse Monitoring for the Marvel Characters model.
It sets up monitoring tables for tracking inference data, data quality, and model drift.

Usage:
    python refresh_monitor.py --root_path <path> --env <environment>

Arguments:
    --root_path: Root path to the project files
    --env: Environment name (dev, staging, prod)
"""

import argparse
import sys

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger

from marvel_characters.config import ProjectConfig
from marvel_characters.monitoring import create_or_refresh_monitoring


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Create or refresh Lakehouse Monitoring for Marvel Characters model"
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


def refresh_monitoring(config_path: str, env: str) -> None:
    """
    Create or refresh monitoring for the deployed model.
    
    Args:
        config_path: Path to the project configuration YAML file
        env: Environment name (dev, staging, prod)
    """
    logger.info("=" * 80)
    logger.info("Marvel Characters Model Monitoring Refresh")
    logger.info("=" * 80)
    
    try:
        # Load project configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = ProjectConfig.from_yaml(config_path=config_path, env=env)
        logger.info(f"✓ Configuration loaded for environment: {env}")
        logger.info(f"Catalog: {config.catalog_name}.{config.schema_name}")
        
        # Initialize Databricks session
        logger.info("\nInitializing Databricks session...")
        spark = DatabricksSession.builder.getOrCreate()
        logger.info(f"✓ Databricks session initialized: {spark.version}")
        
        # Initialize Workspace client
        logger.info("Initializing Workspace client...")
        workspace = WorkspaceClient()
        logger.info("✓ Workspace client initialized")
        
        # Create or refresh monitoring
        logger.info("\nCreating/refreshing monitoring tables...")
        logger.info("This will set up:")
        logger.info("  - Inference table for predictions")
        logger.info("  - Profile metrics table")
        logger.info("  - Drift metrics table")
        logger.info("  - Data quality monitoring dashboards")
        
        create_or_refresh_monitoring(
            config=config,
            spark=spark,
            workspace=workspace
        )
        
        logger.info("✓ Monitoring tables created/refreshed successfully")
        
        logger.info("=" * 80)
        logger.info("Monitoring Refresh Completed Successfully")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info(f"  - Environment: {env}")
        logger.info(f"  - Catalog: {config.catalog_name}.{config.schema_name}")
        logger.info("  - Monitoring tables: Created/Updated")
        logger.info("  - Drift detection: Enabled")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("  1. Review monitoring dashboard in Databricks")
        logger.info("  2. Configure alert thresholds for drift detection")
        logger.info("  3. Schedule periodic monitoring refreshes")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Monitoring Refresh Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


def main() -> None:
    """Main execution function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        logger.info(f"Starting monitoring refresh with root_path={args.root_path}, env={args.env}")
        
        # Build configuration path
        config_path = f"{args.root_path}/files/project_config_marvel.yml"
        
        # Refresh monitoring
        refresh_monitoring(config_path=config_path, env=args.env)
        
        sys.exit(0)
        
    except Exception as e:
        logger.exception(f"Monitoring refresh script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
