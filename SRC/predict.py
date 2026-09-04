from joblib import load
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "Model" / "model.joblib"

model=load(model_path)

prediction=model.predict([[9.5]])
print("the predicted value is:",prediction)