from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote the latest version of "RandomForest" to Staging
client.transition_model_version_stage(
    name="RandomForest",
    version=2,   # version number assigned automatically
    stage="Staging"
)