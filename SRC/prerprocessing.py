import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data=pd.read_csv(r'D:\Ai_Ml _projects\ML_OPS\DATA\placement_RAW_DATA.csv')


    print("the data is loaded sucessfully")
    print(data)
    return data


def preprocess_data(data):
    X=data[['cgpa']]
    y=data['package']

    print("the data is preprocessed sucessfully")
    return X,y

