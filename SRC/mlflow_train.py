import mlflow

from prerprocessing import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

from mlflow.tracking import MlflowClient


# Connect to our MLflow database
mlflow.set_tracking_uri(
    "sqlite:///D:/Ai_Ml_projects/ML_OPS/mlflow.db"
)

# Create/select experiment
mlflow.set_experiment("package experiment")


# Start MLflow run
with mlflow.start_run():

    # Load data
    data = load_data()

    # Preprocess
    X, y = preprocess_data(data)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    # Train
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    rmse = root_mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # Parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("random_state", 42)

    # Metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    plt.scatter(y_test, prediction)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Prediction Scatter Plot")
    plt.savefig("scatter_plot.png")
    mlflow.log_artifact("scatter_plot.png")

     # ⭐ Log trained model
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="RandomForest" 
    )


    

    client = MlflowClient()

    # Promote the latest version of "RandomForest" to Staging
    client.transition_model_version_stage(
        name="RandomForest",
        version=2,   # version number assigned automatically
        stage="Staging"
    )