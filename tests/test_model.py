from pathlib import Path
from joblib import load


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Model" / "model.joblib"


def test_model_prediction():

    model = load(MODEL_PATH)

    prediction = model.predict([[8.0]])

    assert prediction is not None
    assert len(prediction) == 1