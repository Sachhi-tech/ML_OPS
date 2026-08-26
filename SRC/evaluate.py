from train import Model,train_model
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error,r2_score


def Evaluate():
    X_train,X_test,y_train,y_test=train_model()
    model=Model()

    pred=model.predict(X_test)
    score=model.score(X_test,y_test)

    values=dict(zip(('Mean Squared Error','Mean Absolute Error','Root Mean Squared Error','R-squared Score','Model Score'),(round(mean_squared_error(y_test, pred)*100, 2),round(mean_absolute_error(y_test, pred)*100, 2),round(root_mean_squared_error(y_test, pred)*100, 2),round(r2_score(y_test, pred)*100, 2),round(score*100, 2))))

    return values

print(Evaluate())