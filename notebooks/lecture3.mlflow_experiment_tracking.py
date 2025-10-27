# Databricks notebook source
"""MLflow Experiment Tracking Demo.

This notebook demonstrates MLflow capabilities including:
- Experiment creation and management
- Run tracking with parameters, metrics, and artifacts
- Artifact logging (files, figures, images, dictionaries)
- Run searching and filtering
- Nested runs for hyperparameter tuning
"""

import json
import os
from pathlib import Path
from time import time

import mlflow
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.
    
    :return: True if running in Databricks, False otherwise
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def setup_mlflow_tracking() -> str:
    """Configure MLflow tracking URI based on environment.
    
    :return: The configured tracking URI
    """
    if not is_databricks():
        load_dotenv()
        profile = os.environ.get("PROFILE", "DEFAULT")
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
        print(f"MLflow configured for profile: {profile}")
    else:
        print("Running in Databricks environment")
    
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Tracking URI: {tracking_uri}")
    return tracking_uri

# COMMAND ----------
# Environment Setup

tracking_uri = setup_mlflow_tracking()

# COMMAND ----------
# Experiment Creation and Management

EXPERIMENT_NAME = "/Shared/marvel-demo"
REPOSITORY_NAME = "marvelousmlops/marvel-characters"
ARTIFACTS_DIR = Path("../demo_artifacts")

# Create artifacts directory if it doesn't exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    # Create or get existing experiment
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.set_experiment_tags({"repository_name": REPOSITORY_NAME})
    
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Experiment Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    
    # Save experiment details for reference
    with open(ARTIFACTS_DIR / "mlflow_experiment.json", "w") as json_file:
        json.dump(experiment.__dict__, json_file, indent=4)
    print(f"Experiment details saved to {ARTIFACTS_DIR / 'mlflow_experiment.json'}")
    
except Exception as e:
    print(f"Error setting up experiment: {e}")
    raise

# COMMAND ----------
# Retrieve and Search Experiments

# Get experiment by ID
retrieved_experiment = mlflow.get_experiment(experiment.experiment_id)
print(f"Retrieved experiment: {retrieved_experiment.name}")

# Search for experiments with specific tags
experiments = mlflow.search_experiments(
    filter_string=f"tags.repository_name='{REPOSITORY_NAME}'"
)
print(f"\nFound {len(experiments)} experiment(s) with repository tag:")
for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# COMMAND ----------
# Basic Run Management - Simple Example

print("Starting a simple run to demonstrate lifecycle...")
with mlflow.start_run(run_name="simple-demo-run") as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Run is active: {mlflow.active_run() is not None}")

# Verify run ended
print(f"Run ended successfully: {mlflow.active_run() is None}")

# COMMAND ----------
# Advanced Run with Parameters, Metrics, and Tags

GIT_SHA = "1234567890abcd"

print("Starting advanced run with comprehensive logging...")
with mlflow.start_run(
    run_name="marvel-demo-run",
    tags={"git_sha": GIT_SHA},
    description="Marvel character prediction demo run with params and metrics",
) as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    
    # Log parameters (immutable once set)
    mlflow.log_params({
        "type": "marvel_demo",
        "model_type": "classifier",
        "framework": "scikit-learn"
    })
    print("✓ Parameters logged")
    
    # Log metrics
    mlflow.log_metrics({
        "metric1": 1.0,
        "metric2": 2.0,
        "accuracy": 0.85
    })
    print("✓ Metrics logged")

print(f"Run completed: {mlflow.active_run() is None}")

# COMMAND ----------
# Retrieve and Inspect Run Details

print("Retrieving run information...")
run_info = mlflow.get_run(run_id=run_id).to_dictionary()

print(f"\nRun Details:")
print(f"  Name: {run_info['info']['run_name']}")
print(f"  Status: {run_info['info']['status']}")
print(f"  Start Time: {run_info['info']['start_time']}")
print(f"  Duration: {(run_info['info']['end_time'] - run_info['info']['start_time']) / 1000:.2f}s")

# Save run info for reference
with open(ARTIFACTS_DIR / "run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)
print(f"\n✓ Run details saved to {ARTIFACTS_DIR / 'run_info.json'}")

# Display logged data
print("\nLogged Metrics:")
for key, value in run_info["data"]["metrics"].items():
    print(f"  {key}: {value}")

print("\nLogged Parameters:")
for key, value in run_info["data"]["params"].items():
    print(f"  {key}: {value}")

# COMMAND ----------
# Search and Resume Runs

print(f"Searching for runs with git_sha={GIT_SHA}...")
search_results = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string=f"tags.git_sha='{GIT_SHA}'",
)
print(f"Found {len(search_results)} run(s)")

if not search_results.empty:
    found_run_id = search_results.run_id.iloc[0]
    print(f"Resuming run: {found_run_id}")
    
    # Resume the run to add more data
    with mlflow.start_run(run_id=found_run_id) as resumed_run:
        print(f"✓ Run resumed: {resumed_run.info.run_id}")
        
        # Note: Cannot overwrite existing parameters
        # This would fail: mlflow.log_param("type", "marvel_demo2")
        
        # But can add new parameters
        mlflow.log_param(key="purpose", value="get_certified")
        mlflow.log_param(key="environment", value="dev")
        print("✓ Added new parameters to existing run")
else:
    print("No runs found to resume")

# COMMAND ----------
# Advanced Artifact Logging - Multiple Types

print("Starting run with comprehensive artifact logging...")
with mlflow.start_run(
    run_name="marvel-demo-run-artifacts",
    tags={"git_sha": GIT_SHA, "artifact_demo": "true"},
    description="Marvel demo run showcasing various artifact types",
) as artifact_run:
    
    # Log single metric
    mlflow.log_metric(key="metric3", value=3.0)
    
    # Log metric over multiple steps (e.g., training epochs)
    print("\nLogging metrics across training steps:")
    for step in range(3):
        metric_value = 3.0 + step / 2
        mlflow.log_metric(key="training_loss", value=metric_value, step=step)
        print(f"  Step {step}: training_loss={metric_value:.2f}")
    
    # Log text content
    mlflow.log_text("Hello, MLflow! This is a demo text artifact.", "hello.txt")
    print("✓ Text artifact logged")
    
    # Log dictionary as JSON
    mlflow.log_dict(
        {
            "model_config": {"layers": 3, "units": 128, "activation": "relu"},
            "training": {"epochs": 10, "batch_size": 32},
            "status": "success"
        },
        "config_example.json"
    )
    print("✓ Dictionary artifact logged")
    
    # Log directory of artifacts (if exists and not empty)
    if ARTIFACTS_DIR.exists() and any(ARTIFACTS_DIR.iterdir()):
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="demo_artifacts")
        print(f"✓ Directory artifacts logged from {ARTIFACTS_DIR}")
    
    artifact_run_id = artifact_run.info.run_id
    print(f"\n✓ Artifact run completed: {artifact_run_id}")

# COMMAND ----------
# Log Matplotlib Figures

print("Creating and logging matplotlib figure...")
with mlflow.start_run(run_name="marvel-demo-visualization") as viz_run:
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = [0, 1, 2, 3, 4]
    y = [2, 3, 5, 7, 11]
    ax.plot(x, y, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Sample MLflow Figure")
    ax.grid(True, alpha=0.3)
    
    mlflow.log_figure(fig, "sample_plot.png")
    print("✓ Matplotlib figure logged")
    
    # Log images dynamically (e.g., for monitoring training)
    print("\nLogging image sequence:")
    for step in range(3):
        # Generate random image (simulating model outputs or visualizations)
        image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        mlflow.log_image(image, key="generated_image", step=step)
        print(f"  Step {step}: image logged")
    
    print("✓ Visualization run completed")
    plt.close(fig)

# COMMAND ----------
# Advanced Run Searching with Complex Filters

print("Performing advanced run search...")

# Calculate time threshold (1 hour ago)
time_hour_ago = int(time() - 3600) * 1000

# Search with multiple filter criteria
runs = mlflow.search_runs(
    search_all_experiments=True,  # or use experiment_ids=[] or experiment_names=[]
    order_by=["start_time DESC"],
    filter_string=(
        "status='FINISHED' AND "
        f"start_time>{time_hour_ago} AND "
        "run_name LIKE '%marvel-demo-run%' AND "
        "metrics.metric3>0 AND "
        "tags.mlflow.source.type!='JOB'"
    )
)

print(f"\nFound {len(runs)} runs matching criteria:")
if not runs.empty:
    print(f"Columns: {list(runs.columns[:10])}...")  # Show first 10 columns
    print(f"\nFirst run:")
    print(f"  Run ID: {runs.run_id.iloc[0]}")
    print(f"  Name: {runs['tags.mlflow.runName'].iloc[0] if 'tags.mlflow.runName' in runs.columns else 'N/A'}")
else:
    print("No runs matched the search criteria")

# Display the results
runs

# COMMAND ----------
# Load and Work with Artifacts

if not runs.empty:
    print("Loading artifacts from the most recent run...")
    artifact_uri = runs.artifact_uri.iloc[0]
    print(f"Artifact URI: {artifact_uri}")
    
    try:
        # Load dictionary artifact
        loaded_dict = mlflow.artifacts.load_dict(f"{artifact_uri}/config_example.json")
        print(f"\n✓ Loaded dictionary: {loaded_dict}")
    except Exception as e:
        print(f"Could not load dictionary: {e}")
    
    try:
        # Load image artifact
        loaded_image = mlflow.artifacts.load_image(f"{artifact_uri}/sample_plot.png")
        print(f"✓ Loaded image shape: {loaded_image.size if hasattr(loaded_image, 'size') else 'N/A'}")
    except Exception as e:
        print(f"Could not load image: {e}")
    
    try:
        # Download artifacts to local directory
        download_path = "../downloaded_artifacts"
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"{artifact_uri}/demo_artifacts",
            dst_path=download_path
        )
        print(f"✓ Artifacts downloaded to {download_path}")
    except Exception as e:
        print(f"Could not download artifacts: {e}")
else:
    print("No runs available to load artifacts from")

# COMMAND ----------
# Nested Runs - Hyperparameter Tuning Pattern

print("Demonstrating nested runs for hyperparameter tuning...\n")

with mlflow.start_run(run_name="marvel_hyperparameter_tuning") as parent_run:
    print(f"Parent run ID: {parent_run.info.run_id}")
    
    # Log parent-level parameters
    mlflow.log_params({
        "optimization_strategy": "grid_search",
        "total_trials": 4
    })
    
    # Simulate hyperparameter tuning with nested runs
    best_metric = float('-inf')
    best_run_id = None
    
    for i in range(1, 5):
        with mlflow.start_run(
            run_name=f"marvel_trial_{i}",
            nested=True,
            tags={"trial_number": str(i)}
        ) as child_run:
            # Simulate different hyperparameters
            learning_rate = 0.001 * (2 ** i)
            batch_size = 16 * i
            
            mlflow.log_params({
                "learning_rate": learning_rate,
                "batch_size": batch_size
            })
            
            # Simulate metrics (normally from model training)
            accuracy = 0.75 + i * 0.03
            loss = 1.5 - i * 0.2
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "loss": loss,
                "f1_score": 0.7 + i * 0.04
            })
            
            print(f"Trial {i}: lr={learning_rate:.4f}, batch={batch_size}, accuracy={accuracy:.4f}")
            
            # Track best run
            if accuracy > best_metric:
                best_metric = accuracy
                best_run_id = child_run.info.run_id
    
    # Log best results to parent run
    mlflow.log_params({"best_child_run_id": best_run_id})
    mlflow.log_metrics({"best_accuracy": best_metric})
    
    print(f"\n✓ Hyperparameter tuning completed")
    print(f"  Best accuracy: {best_metric:.4f}")
    print(f"  Best run ID: {best_run_id}")

# COMMAND ----------
print("="*60)
print("MLflow Experiment Tracking Demo Completed Successfully!")
print("="*60)
print("\nKey Takeaways:")
print("  • Experiments organize related runs")
print("  • Runs track parameters, metrics, and artifacts")
print("  • Parameters are immutable, metrics can be logged multiple times")
print("  • Artifacts support various types: files, figures, images, dicts")
print("  • Search enables filtering and retrieving runs")
print("  • Nested runs are useful for hyperparameter tuning")
print("="*60)

# COMMAND ----------
