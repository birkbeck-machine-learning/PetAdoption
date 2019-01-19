import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def FFWWithKFold(Inputs, Targets, num_units, num_outputs, learning_rate, num_epochs):

    mms = MinMaxScaler()
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
        mms.fit(X_train)
        X_train = mms.transform(X_train)
        X_test = mms.transform(X_test)

        # Construction Phase
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.float32, shape=(None), name="y")

        training = tf.placeholder_with_default(False, shape=(), name='training')
        dropout_rate = 0.5  # 1 - keep_prob

        hidden_layers = []

        with tf.name_scope("dnn"):

            for i, n in enumerate(num_units):
                if(i == 0):
                    hidden_layers.append(
                        tf.layers.dropout(tf.layers.dense(X, n, name="hidden" + str(i), activation=tf.nn.relu),
                                          dropout_rate, training=training))
                else:
                    hidden_layers.append(
                        tf.layers.dropout(tf.layers.dense(hidden_layers[i-1], n, name="hidden" + str(i), activation=tf.nn.relu),
                                          dropout_rate, training=training))

            logits = tf.layers.dense(hidden_layers[-1], num_outputs, name="outputs")

        with tf.name_scope("loss"):
            error = logits - y
            mse = tf.reduce_mean(tf.square(error), name="mse")

        with tf.name_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(mse)

        init = tf.global_variables_initializer()

        # Execution Phase """
        with tf.Session() as sess:
            init.run()

            for epoch in range(num_epochs):
                if(epoch % 100 == 0):
                    print('Epoch: ' + str(epoch) + ' / ' + str(num_epochs))
                sess.run(training_op, feed_dict={X: X_train, y: y_train, training: True})

                TrainingErrors[fold_num].append(mse.eval(feed_dict={X: X_train, y: y_train, training: False}))

            TrainingErrors.append([])
            TestErrors.append(mse.eval(feed_dict={X: X_test, y: y_test, training: False}))

        fold_num += 1

    TrainingErrors.pop(-1)

    return TrainingErrors, TestErrors

