import numpy as np
import pandas as pd
import os
import cv2
import glob

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

def GetImageTrainingData(im_width, im_height):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv(dir_path + "/input/train/train.csv")

    im_dict = {'Image': [], 'AdoptionSpeed': []}
    files = os.path.dirname(os.path.realpath(__file__)) + '/input/train_images/*.jpg'
    files = glob.glob(files)

    for i, filename in enumerate(files):
        image = cv2.imread(filename)
        image = cv2.resize(image, (im_width, im_height))
        im_dict['Image'].append(image)

        ID = filename.split('\\')[-1].split('-')[0]
        ind = df_train.index[df_train['PetID'] == ID].tolist()[0]
        adop_speed = df_train.iloc[ind]['AdoptionSpeed']
        im_dict['AdoptionSpeed'].append(adop_speed)

        if(i % 1000 == 0):
            print('Processed image ' + str(i) + ' / ' + str(files.__len__()))

    return np.array(im_dict['Image']), np.array(im_dict['AdoptionSpeed'])