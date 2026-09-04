from prerprocessing import preprocess_data,load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

def train_model():
    data=load_data()
    X,y=preprocess_data(data)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    return X_train,X_test,y_train,y_test 

def Model():
    X_train,_,y_train,_=train_model()
    model=LinearRegression()
    model.fit(X_train,y_train)

    return model


def Save_model(model):
    f_model=dump(model,
                 r"D:\Ai_Ml _projects\ML_OPS\Model\model.joblib"
                 )

    print("the model is saved sucessfully")


if __name__=="__main__":
    model=Model()
    Save_model(model)