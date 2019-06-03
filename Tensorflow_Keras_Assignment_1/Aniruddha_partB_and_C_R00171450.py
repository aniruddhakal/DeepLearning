import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import h5py

np.random.seed(171450)
tf.set_random_seed(171450)


def load_data(h5_file):
    with h5py.File(h5_file, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())

        allTrain = hf.get('trainData')
        allTest = hf.get('testData')

        npTrain = np.array(allTrain)
        npTest = np.array(allTest)

        print('Shape of the array dataset_1: \n', npTrain.shape)
        print('Shape of the array dataset_2: \n', npTest.shape)

    return npTrain[:, :-1], npTrain[:, -1], npTest[:, :-1], npTest[:, -1]


def plot_graph(history, n_epochs):
    """
    this plot code is copied from Week 9 lecture slides of 'Deep Learning'
    """
    plt.style.use("ggplot")
    plt.figure(dpi=300)
    plt.plot(np.arange(0, n_epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def get_default_config():
    model = tf.keras.models.Sequential()

    # output layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model


def get_configuration1():
    model = tf.keras.models.Sequential()

    # H1 layer
    model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
    # output layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model


def get_configuration2(INCLUDE_DROPOUT, INCLUDE__L1_REGULARIZATION, INCLUDE__L2_REGULARIZATION):
    model = tf.keras.models.Sequential()

    h1_regularizer = None
    h2_regularizer = None

    h1_l1 = 0.
    h1_l2 = 0.
    h2_l1 = 0.
    h2_l2 = 0.

    if INCLUDE__L1_REGULARIZATION:
        print("Applying L1 Regularization")
        h1_l1 = 0.0001
        h2_l1 = 0.00001

    if INCLUDE__L2_REGULARIZATION:
        print("Applying L2 Regularization")
        h1_l2 = 0.0001
        h2_l2 = 0.00001

    if INCLUDE__L1_REGULARIZATION or INCLUDE__L2_REGULARIZATION:
        h1_regularizer = tf.keras.regularizers.l1_l2(l1=h1_l1, l2=h1_l2)
        h2_regularizer = tf.keras.regularizers.l1_l2(l1=h2_l1, l2=h2_l2)

    INCLUDE_DROPOUT and print("Applying dropout")

    # H1 layer
    model.add(tf.keras.layers.Dense(400, activation=tf.nn.relu,
                                    kernel_regularizer=h1_regularizer))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.4))

    # H2 layer
    model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu,
                                    kernel_regularizer=h2_regularizer))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))

    # output layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model


def get_configuration3(INCLUDE_DROPOUT, INCLUDE__L1_REGULARIZATION, INCLUDE__L2_REGULARIZATION):
    h1_regularizer = None
    h2_regularizer = None
    h3_regularizer = None

    h1_l1 = 0.
    h1_l2 = 0.
    h2_l1 = 0.
    h2_l2 = 0.
    h3_l1 = 0.
    h3_l2 = 0.

    if INCLUDE__L1_REGULARIZATION:
        print("Applying L1 Regularization")
        h1_l1 = 0.001
        h2_l1 = 0.0001
        h3_l1 = 0.00001

    if INCLUDE__L2_REGULARIZATION:
        print("Applying L2 Regularization")
        h1_l2 = 0.001
        h2_l2 = 0.0001
        h3_l2 = 0.00001

    INCLUDE_DROPOUT and print("Applying dropout")

    if INCLUDE__L1_REGULARIZATION or INCLUDE__L2_REGULARIZATION:
        h1_regularizer = tf.keras.regularizers.l1_l2(l1=h1_l1, l2=h1_l2)
        h2_regularizer = tf.keras.regularizers.l1_l2(l1=h2_l1, l2=h2_l2)
        h3_regularizer = tf.keras.regularizers.l1_l2(l1=h3_l1, l2=h3_l2)

    model = tf.keras.models.Sequential()

    # H1 layer
    model.add(tf.keras.layers.Dense(600, activation=tf.nn.relu,
                                    kernel_regularizer=h1_regularizer))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.072))

    # H2 layer
    model.add(tf.keras.layers.Dense(400, activation=tf.nn.relu,
                                    kernel_regularizer=h2_regularizer))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.4))

    # H3 layer
    model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu,
                                    kernel_regularizer=h3_regularizer))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))

    # output layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model


def main(configuration, INCLUDE_DROPOUT=False, INCLUDE__L1_REGULARIZATION=False,
         INCLUDE__L2_REGULARIZATION=False):
    """
    TODO references to cite:
    :return:
    """

    # TODO set epochs to 30
    n_epochs = 20
    learning_rate = 0.001

    # batch size 256 as stated in task(i)
    batch_size = 256
    validation_split_ratio = 0.1
    random_seed = 171450

    tf.random.set_random_seed(random_seed)

    data_dir = "./"
    file = "data.h5"
    h5_file = data_dir + file

    trainX, trainY, testX, testY = load_data(h5_file)

    trainX = tf.keras.utils.normalize(trainX, axis=1)
    testX = tf.keras.utils.normalize(testX, axis=1)

    # TODO take visualization report for all 3 configurations
    # TODO add functions for each configuration
    # TODO apply regularization, L1, L2 and Dropout, etc, check assignment description

    # based on Boolean REGULARIZATION_FLAG, load respective NN configurations
    if configuration == 'config1':
        model = get_configuration1()
    elif configuration is 'config2':
        model = get_configuration2(INCLUDE_DROPOUT, INCLUDE__L1_REGULARIZATION, INCLUDE__L2_REGULARIZATION)
    elif configuration is 'config3':
        model = get_configuration3(INCLUDE_DROPOUT, INCLUDE__L1_REGULARIZATION, INCLUDE__L2_REGULARIZATION)
    else:
        model = get_default_config()

    model.compile(  # optimizer="adam",
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])

    fit_start_time = time()
    history = model.fit(trainX, trainY, batch_size=batch_size, epochs=n_epochs,
                        validation_split=validation_split_ratio)
    fit_time = time() - fit_start_time

    prediction_start_time = time()


    # results = model.evaluate(testX, testY)
    prediction_results = model.predict(testX)

    results = np.argmax(prediction_results, axis=1)
    prediction_time = time() - prediction_start_time

    print("\n\nConfusion Matrix: \n", confusion_matrix(testY, results))
    print("\n\n", classification_report(testY, results))
    print("\n\nAccuracy Score: ", accuracy_score(testY, results) * 100)

    print("\n\nHistory:\n", history.history.keys())
    print("\n\nHistory Values:\nVal_acc:", history.history['val_acc'], "\nVal_Loss: ",
          history.history['val_loss'],
          "\nAcc: ", history.history['acc'], "\nLoss: ", history.history['loss'])

    print("\n\nTime Taken: Fit Time: %f seconds" % fit_time,
          ", Prediction Time: %f seconds" % prediction_time)

    plot_graph(history, n_epochs)


if __name__ == '__main__':
    # Adjust Dropout and Regularization settings from here
    INCLUDE_DROPOUT = False
    INCLUDE__L1_REGULARIZATION = False
    INCLUDE__L2_REGULARIZATION = False

    # uncomment only 1 of the below configurations to run it
    # configuration = 'default' # this is the baseline regression config
    # configuration = 'config1'
    # configuration = 'config2'
    configuration = 'config3'

    main(configuration, INCLUDE_DROPOUT, INCLUDE__L1_REGULARIZATION, INCLUDE__L2_REGULARIZATION)
