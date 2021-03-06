import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

from Parsers import GetImageTrainingData
from ImageAnalysis.NeuralNetworks.Conv import TrainConvWithKFold

im_width = 84
im_height = 84
X_train, y_train = GetImageTrainingData(im_width, im_height)

print('Number of Examples: ' + str(X_train.shape[0]))

dropout_rates = [0, .1, .2, .3, .4]
filters = [32, 64, 64]
kernels = [8, 4, 3]
strides = [4, 2, 1]
dense_units = [128, 64]
num_outputs = 1
learning_rate = .0001
num_epochs = 10000
batch_size = 32

training_results = []
test_results = []

for dropout_rate in dropout_rates:

    print('Cross Validating Dropout Rate: ' + str(dropout_rate))
    training_error, test_error = TrainConvWithKFold(X_train, y_train, im_width, im_height,
                                                    filters, kernels, strides, dense_units,
                                                    num_outputs, learning_rate, num_epochs,
                                                    batch_size, dropout_rate)

    training_results.append(training_error)
    test_results.append(test_error)


colours = ['r', 'b', 'c', 'k', 'g']

# Training Curves
plt.figure()
for i in range(dropout_rates.__len__()):

    x = np.arange(num_epochs)
    y = np.mean(training_results[i], axis=0)
    error = np.std(training_results[i], axis=0)

    plt.plot(x, y, color=colours[i], label='Dropout Rate: ' + str(dropout_rates[i]))
    plt.fill_between(x, y-error, y+error, alpha=.5, color=colours[i])

plt.legend()
plt.ylabel('Mean Squared Error')
plt.xlabel('Training Epoch')
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Plots/Conv_Training_Curves.png'))
plt.close()

# Training and Test Error
plt.figure()
for i in range(dropout_rates.__len__()):

    values = np.array(training_results[i])[:, -1]
    plt.errorbar(dropout_rates[i], np.mean(values), yerr=np.std(values), marker='o',
                 color='r', label='Training MSE', alpha=.5)
    values = np.array(test_results[i])
    plt.errorbar(dropout_rates[i], np.mean(values), yerr=np.std(values), marker='o',
                 color='b', label='Test MSE', alpha=.5)


plt.ylabel('Mean Squared Error')
plt.xlabel('Dropout Rate')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.legend(by_label.values(), by_label.keys())
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Plots/Conv_Training_And_Test_Error.png'))
plt.close()
