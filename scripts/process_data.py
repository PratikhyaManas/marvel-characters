import argparse
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config_marvel.yml"


config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the Marvel characters dataset
spark = SparkSession.builder.getOrCreate()

# Example: Adjust the path and loading logic as per your Marvel dataset location
# This assumes the Marvel dataset is stored as a Delta table in the configured catalog/schema
marvel_table = f"{config.catalog_name}.{config.schema_name}.marvel_characters"
df = spark.table(marvel_table).toPandas()

# If you have Marvel-specific synthetic/test data generation, use them here.
# Otherwise, just use the loaded Marvel dataset as is.
new_data = df.copy()
logger.info("Marvel data loaded for processing.")

# Initialize DataProcessor
data_processor = DataProcessor(new_data, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
