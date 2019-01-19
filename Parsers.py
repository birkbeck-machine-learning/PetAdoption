import numpy as np
import pandas as pd
import os

from enum import Enum


class DataType(Enum):
    nominal = 1
    ordinal = 2
    continous = 3


def GetTrainingData():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv(dir_path + "/input/train/train.csv")

    X_train = df_train.drop(['AdoptionSpeed'], axis=1)
    y_train = df_train['AdoptionSpeed'].to_frame()

    return X_train, y_train


def GetNumericalTrainingData():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv(dir_path + "/input/train/train.csv")

    X_train = df_train.drop(['AdoptionSpeed'], axis=1)
    y_train = df_train['AdoptionSpeed'].to_frame()

    # Separate the training data based on whether it is numerical or not
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numerical = X_train[numerical_cols]

    data_type_dict = {'Type': DataType.nominal,
                      'Age': DataType.continous,
                      'Breed1': DataType.nominal,
                      'Breed2': DataType.nominal,
                      'Gender': DataType.nominal,
                      'Color1': DataType.nominal,
                      'Color2': DataType.nominal,
                      'Color3': DataType.nominal,
                      'MaturitySize': DataType.ordinal,
                      'FurLength': DataType.ordinal,
                      'Vaccinated': DataType.nominal,
                      'Dewormed': DataType.nominal,
                      'Sterilized': DataType.nominal,
                      'Health': DataType.ordinal,
                      'Quantity': DataType.continous,
                      'Fee': DataType.continous,
                      'State': DataType.nominal,
                      'VideoAmt': DataType.continous,
                      'PhotoAmt': DataType.continous}


    return X_train_numerical, y_train, data_type_dict


def GetTextTrainingData():
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv(dir_path + "/input/train/train.csv")

    X_train = df_train.drop(['AdoptionSpeed'], axis=1)
    y_train = df_train['AdoptionSpeed'].to_frame()

    # Separate the training data based on whether it is numerical or not
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_text = X_train.drop(numerical_cols, axis=1)

    return X_train_text, y_train