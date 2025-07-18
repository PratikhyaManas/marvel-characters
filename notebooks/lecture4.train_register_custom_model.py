# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import MarvelModelWrapper
from importlib.metadata import version
from dotenv import load_dotenv

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    import os
    os.environ["PROFILE"] = "marvelous"
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})
marvel_characters_v = version("marvel_characters")

code_paths=[f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]

# COMMAND ----------
wrapped_model_version = get_model_version_by_alias(
    name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic",
    alias="latest-model")
# Initialize model with the config path

# COMMAND ----------
wrapper = MarvelModelWrapper()
wrapper.log_register_model(wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
                           pyfunc_model_name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_pyfunc",
                           experiment_name=config.experiment_name_custom,
                           tags=tags,
                           code_paths=code_paths)

# COMMAND ----------
loaded_pufunc_model = mlflow.pyfunc.load_model(pyfunc_model.model_uri)
# COMMAND ----------
client.set_registered_model_alias(
    name=registered_model_name,
    alias="latest-model",
    version=pyfunc_model.registered_model_version,
)
# COMMAND ----------
unwraped_model = loaded_pufunc_model.unwrap_python_model()
unwraped_model.predict(context=None, model_input=X_test[0:1])

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/marvel-characters-custom"], filter_string="tags.branch='module2'"
).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-marvel-character-model")

# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ---------- 