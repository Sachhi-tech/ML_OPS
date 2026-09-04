import mlflow
import mlflow.sklearn


# Connect to the same MLflow database
mlflow.set_tracking_uri(
    "sqlite:///D:/Ai_Ml _projects/ML_OPS/mlflow.db"
)


# Run ID of the model we want to load
run_id = "64b185a1a849483aa440c5f8bec58657"


# Model location inside that run
model_uri = "models:/RandomForest@champion"


# Load the model
model = mlflow.sklearn.load_model(model_uri)


# Make prediction
cgpa = [[8.0]]

prediction = model.predict(cgpa)


print("CGPA:", cgpa[0][0])
print("Predicted Package:", prediction[0])