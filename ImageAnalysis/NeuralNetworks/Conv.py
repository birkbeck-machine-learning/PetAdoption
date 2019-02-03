import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def TrainConvWithKFold(Inputs, Targets, im_width, im_height, filters, kernels, strides, dense_units, num_outputs, learning_rate, num_epochs, batch_size, dropout_rate):

    kf = KFold(n_splits=5)

    TrainingErrors = [[]]
    TestErrors = []
    n_inputs = Inputs.shape[1]
    fold_num = 0

    for train_index, test_index in kf.split(Inputs):

        print('Starting Fold Number: ' + str(fold_num))

        # Get training and test sets
        X_train, X_test = Inputs[train_index], Inputs[test_index]
        y_train, y_test = Targets[train_index], Targets[test_index]

        # Scale inputs
        X_train = X_train / 255
        X_test = X_test / 255

        # Construction Phase
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, shape=(None, im_width, im_height, 3), name="X")
        y = tf.placeholder(tf.float32, shape=(None), name="y")

        training = tf.placeholder_with_default(False, shape=(), name='training')

        conv_layers = []
        dense_layers = []

        with tf.name_scope("dnn"):

            for i in range(kernels.__len__()):
                if (i == 0):
                    conv_layers.append(
                        tf.layers.conv2d(inputs=X, filters=filters[i], kernel_size=[kernels[i], kernels[i]],
                                         strides=strides[i], padding="same",
                                         activation=tf.nn.relu, name="conv" + str(i)))
                else:
                    conv_layers.append(
                        tf.layers.conv2d(inputs=conv_layers[i - 1], filters=filters[i], kernel_size=[kernels[i], kernels[i]],
                                         strides=strides[i], padding="same",
                                         activation=tf.nn.relu, name="conv" + str(i)))


            conv_flat = tf.reshape(conv_layers[-1], [-1, conv_layers[-1].shape[1] * conv_layers[-1].shape[2] * filters[-1]])

            for i, n in enumerate(dense_units):
                if (i == 0):
                    dense_layers.append(
                        tf.layers.dropout(
                            tf.layers.dense(conv_flat, n, name="dense" + str(i), activation=tf.nn.relu),
                            dropout_rate, training=training))
                else:
                    dense_layers.append(
                        tf.layers.dropout(
                            tf.layers.dense(dense_layers[i - 1], n, name="dense" + str(i), activation=tf.nn.relu),
                            dropout_rate, training=training))


            logits = tf.layers.dense(dense_layers[-1], num_outputs, name="outputs")


        with tf.name_scope("loss"):
            error = logits - y
            mse = tf.reduce_mean(tf.square(error), name="mse")

        with tf.name_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(mse)

        init = tf.global_variables_initializer()

        # Execution Phase
        with tf.Session() as sess:
            init.run()

            for epoch in range(num_epochs):
                if(epoch % 100 == 0):
                    print('Epoch: ' + str(epoch) + ' / ' + str(num_epochs))

                inds = np.random.randint(0, X_train.shape[0], batch_size)
                sess.run(training_op, feed_dict={X: X_train[inds], y: y_train[inds], training: True})

                inds = np.random.randint(0, X_train.shape[0], 1000)
                TrainingErrors[fold_num].append(mse.eval(feed_dict={X: X_train[inds], y: y_train[inds], training: False}))

            TrainingErrors.append([])

            inds = np.random.randint(0, X_test.shape[0], 1000)
            TestErrors.append(mse.eval(feed_dict={X: X_test[inds], y: y_test[inds], training: False}))

        fold_num += 1

    TrainingErrors.pop(-1)

    return TrainingErrors, TestErrors

