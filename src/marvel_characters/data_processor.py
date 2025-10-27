"""Data preprocessing module for Marvel characters."""

import time
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from marvel_characters.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing Marvel character DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        """
        Initialize DataProcessor with data and configuration.
        
        Args:
            pandas_df: Input DataFrame with Marvel characters data
            config: Project configuration object
            spark: Spark session instance
        """
        self.df = pandas_df
        self.config = config
        self.spark = spark
        logger.debug(f"DataProcessor initialized with {len(pandas_df)} rows")

    def preprocess(self) -> None:
        """Preprocess the Marvel character DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        logger.info("Starting data preprocessing...")
        initial_rows = len(self.df)
        
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target

        # Rename columns for consistency
        self.df.rename(columns={"Height (m)": "Height", "Weight (kg)": "Weight"}, inplace=True)
        logger.debug("Column names standardized")

        # Universe preprocessing
        self.df["Universe"] = self.df["Universe"].fillna("Unknown")
        counts = self.df["Universe"].value_counts()
        small_universes = counts[counts < 50].index
        self.df["Universe"] = self.df["Universe"].replace(small_universes, "Other")
        logger.debug(f"Universe feature processed: {len(small_universes)} small universes grouped as 'Other'")

        # Teams - convert to binary indicator
        self.df["Teams"] = self.df["Teams"].notna().astype("int")

        # Origin preprocessing
        self.df["Origin"] = self.df["Origin"].fillna("Unknown")

        # Identity preprocessing
        self.df["Identity"] = self.df["Identity"].fillna("Unknown")
        self.df = self.df[self.df["Identity"].isin(["Public", "Secret", "Unknown"])]

        # Gender preprocessing
        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Gender"] = self.df["Gender"].where(self.df["Gender"].isin(["Male", "Female"]), other="Other")

        # Marital status preprocessing
        self.df.rename(columns={"Marital Status": "Marital_Status"}, inplace=True)
        self.df["Marital_Status"] = self.df["Marital_Status"].fillna("Unknown")
        self.df["Marital_Status"] = self.df["Marital_Status"].replace("Widow", "Widowed")
        self.df = self.df[self.df["Marital_Status"].isin(["Single", "Married", "Widowed", "Engaged", "Unknown"])]

        # Feature engineering: Magic indicator
        self.df["Magic"] = self.df["Origin"].str.lower().apply(lambda x: int("magic" in x))

        # Feature engineering: Mutant indicator
        self.df["Mutant"] = self.df["Origin"].str.lower().apply(lambda x: int("mutate" in x or "mutant" in x))
        logger.debug("Engineered features created: Magic, Mutant")

        # Normalize origin categories
        def normalize_origin(x):
            x_lower = str(x).lower()
            if "human" in x_lower:
                return "Human"
            elif "mutate" in x_lower or "mutant" in x_lower:
                return "Mutant"
            elif "asgardian" in x_lower:
                return "Asgardian"
            elif "alien" in x_lower:
                return "Alien"
            elif "symbiote" in x_lower:
                return "Symbiote"
            elif "robot" in x_lower:
                return "Robot"
            elif "cosmic being" in x_lower:
                return "Cosmic Being"
            else:
                return "Other"

        self.df["Origin"] = self.df["Origin"].apply(normalize_origin)
        logger.debug("Origin categories normalized")

        # Target variable: Alive status
        self.df = self.df[self.df["Alive"].isin(["Alive", "Dead"])]
        self.df["Alive"] = (self.df["Alive"] == "Alive").astype(int)

        # Select final features
        self.df = self.df[num_features + cat_features + [target] + ["PageID"]]

        # Convert categorical features
        for col in cat_features:
            self.df[col] = self.df[col].astype("category")

        # Rename PageID to Id for consistency
        if "PageID" in self.df.columns:
            self.df = self.df.rename(columns={"PageID": "Id"})
            self.df["Id"] = self.df["Id"].astype("str")

        final_rows = len(self.df)
        logger.info(f"✓ Preprocessing completed: {initial_rows} → {final_rows} rows ({final_rows/initial_rows*100:.1f}% retained)")
        logger.info(f"Features: {len(num_features)} numerical, {len(cat_features)} categorical")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        Args:
            test_size: The proportion of the dataset to include in the test split
            random_state: Controls the shuffling applied to the data before applying the split
            
        Returns:
            Tuple containing the training and test DataFrames
        """
        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        logger.info(f"✓ Data split completed: train={len(train_set)}, test={len(test_set)}")
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        Args:
            train_set: The training DataFrame to be saved
            test_set: The test DataFrame to be saved
        """
        logger.info("Saving datasets to Unity Catalog...")
        
        train_table = f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        test_table = f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        logger.debug(f"Writing train set to {train_table}")
        train_set_with_timestamp.write.mode("overwrite").saveAsTable(train_table)

        logger.debug(f"Writing test set to {test_table}")
        test_set_with_timestamp.write.mode("overwrite").saveAsTable(test_table)
        
        logger.info(f"✓ Datasets saved successfully to {self.config.catalog_name}.{self.config.schema_name}")

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        logger.info("Enabling Change Data Feed for train and test tables...")
        
        train_table = f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        test_table = f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        
        self.spark.sql(
            f"ALTER TABLE {train_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {test_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        
        logger.info("✓ Change Data Feed enabled for both tables")


def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic Marvel character data matching input DataFrame distributions with optional drift.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types. Supports intentional data drift for specific features when enabled.

    Args:
        df: Source DataFrame containing original data distributions
        drift: Flag to activate synthetic data drift injection
        num_rows: Number of synthetic records to generate
        
    Returns:
        DataFrame containing generated synthetic data
    """
    logger.info(f"Generating {num_rows} synthetic records (drift={drift})")
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"Height", "Weight"}:  # Handle physical attributes
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)
                # Ensure positive values for physical attributes
                synthetic_data[column] = np.maximum(0.1, synthetic_data[column])
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to appropriate types
    float_columns = {"Height", "Weight"}
    for col in float_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    # Generate unique IDs based on timestamp
    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        logger.debug("Applying drift to synthetic data...")
        # Skew the physical attributes to introduce drift
        drift_features = ["Height", "Weight"]
        for feature in drift_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

        # Introduce bias in categorical features
        if "Gender" in synthetic_data.columns:
            synthetic_data["Gender"] = np.random.choice(["Male", "Female"], num_rows, p=[0.7, 0.3])
        logger.debug("Drift applied: Height/Weight scaled by 1.5x, Gender biased 70/30")

    logger.info(f"✓ Synthetic data generated: {synthetic_data.shape}")
    return synthetic_data


def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching input DataFrame distributions with optional drift.
    
    Args:
        df: Source DataFrame containing original data distributions
        drift: Flag to activate synthetic data drift injection
        num_rows: Number of test records to generate
        
    Returns:
        DataFrame containing generated test data
    """
    return generate_synthetic_data(df, drift, num_rows)
