from joblib import load



model=load(r"D:\Ai_Ml _projects\ML_OPS\MODEL\model.joblib")

prediction=model.predict([[9.5]])
print("the predicted value is:",prediction)