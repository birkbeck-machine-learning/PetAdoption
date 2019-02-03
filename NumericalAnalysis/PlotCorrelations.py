import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from Parsers import GetNumericalTrainingData, DataType

X_train_numerical, y_train, data_type_dict = GetNumericalTrainingData()

# Examine the variables that have order
cols = [k for k, v in data_type_dict.items() if v == DataType.ordinal or v == DataType.continous]

# See which variables are correlated with adoption speed
corrcoefs = np.zeros((cols.__len__()))

for i, column in enumerate(cols):
    corrcoefs[i] = spearmanr(X_train_numerical[column].values, y_train['AdoptionSpeed'].values)[0]

plt.figure(figsize=(10, 10))
plt.bar(np.arange(corrcoefs.shape[0]), corrcoefs)
plt.xticks(np.arange(corrcoefs.shape[0]), cols, rotation=270)
plt.savefig('Plots/XY_Correlations.png')
plt.close()

# Examine the correlations between input variables
X_corr = X_train_numerical[cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(X_corr.values, interpolation='nearest')
plt.xticks(np.arange(X_corr.shape[0]), cols, rotation=270)
plt.yticks(np.arange(X_corr.shape[0]), cols)
fig.colorbar(cax)
plt.savefig('Plots/XX_Correlations.png')
plt.close()
