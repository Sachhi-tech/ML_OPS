from joblib import load


def test_model_prediction():

    model = load("Model/model.joblib")

    prediction = model.predict([[8.0]])

    assert prediction is not None
    assert len(prediction) == 1