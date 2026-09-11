from fastapi import FastAPI
import numpy as np
from joblib import load
import pandas as pd


model=load("Model/model.joblib")
if model is None:
    raise ValueError("Model not found. Please train the model first.")
else:
    app = FastAPI()

    @app.get("/")
    def home():
        return {"message": "Welcome to the ML model API!"}

    @app.post("/predict/{input_data}")
    def predict(input_data: float):
        try:
            # Convert the input string to a numpy array
            input_data = pd.DataFrame({"cgpa": [input_data]})
            prediction = model.predict(input_data)
            return {"prediction": prediction.tolist()}
        except Exception as e:
            return {"error": str(e)}
