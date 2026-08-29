import mlflow

mlflow.set_experiment("placement_experiment")

with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("feature", "cgpa")

    mlflow.log_metric("r2_score", 0.85)
    mlflow.log_metric("mae", 0.45)

    print("MLflow run completed")