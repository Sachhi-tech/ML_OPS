from sklearn.linear_model import LinearRegression


def test_model_prediction():

    model = LinearRegression()

    model.fit(
        [[1], [2], [3]],
        [2, 4, 6]
    )

    prediction = model.predict([[4]])

    assert prediction is not None
    assert len(prediction) == 1