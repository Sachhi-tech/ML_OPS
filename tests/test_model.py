from joblib import load
from pathlib import Path
from predict import BASE_DIR


def test_model_prediction():

    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "Model" / "model.joblib"

    model = load(model_path)

    prediction = model.predict([[8.0]])

    assert prediction is not None
    assert len(prediction) == 1