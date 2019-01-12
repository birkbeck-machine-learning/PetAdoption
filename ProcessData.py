import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


print(os.listdir("input"))

df_train = pd.read_csv("input/train/train.csv")



X_train = df_train.drop(['AdoptionSpeed'], axis=1)
y_train = df_train['AdoptionSpeed'].to_frame()

print(X_train.head())
print(y_train.head())

# Separate the trianing data based on whether it is numerical or not
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train_numerical = X_train[numerical_cols]
X_train_text = X_train.drop(numerical_cols, axis=1)

from enum import Enum
class DataType(Enum):
    nominal = 1
    ordinal = 2
    continous = 3


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


# Examine the variables that have order
cols = [k for k, v in data_type_dict.items() if v == DataType.ordinal or v == DataType.continous]

# See which variables are correlated with adoption speed
corrcoefs = np.zeros((cols.__len__()))

from scipy.stats import spearmanr

for i, column in enumerate(cols):

    corrcoefs[i] = spearmanr(X_train_numerical[column].values, y_train['AdoptionSpeed'].values)[0]

plt.figure(figsize=(10, 10))
plt.bar(np.arange(corrcoefs.shape[0]), corrcoefs)
plt.xticks(np.arange(corrcoefs.shape[0]), cols, rotation=270)
plt.savefig('plots/Y_Correlations.png')
plt.close()


# Examine the correlations between input variables
X_corr = X_train_numerical[cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(X_corr.values, interpolation='nearest')
plt.xticks(np.arange(X_corr.shape[0]), cols, rotation=270)
plt.yticks(np.arange(X_corr.shape[0]), cols)
fig.colorbar(cax)
plt.savefig('plots/X_Correlations.png')
plt.close()
