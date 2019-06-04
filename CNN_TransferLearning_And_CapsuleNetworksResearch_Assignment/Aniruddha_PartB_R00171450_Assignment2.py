import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import sklearn

import h5py


def plot_graph(history, n_epochs, save_flag=False, target_file=''):
    """
    this plot code is copied from Week 9 lecture slides of 'Deep Learning'
    """
    plt.style.use("ggplot")
    #     plt.figure(dpi=150)
    plt.plot(np.arange(0, n_epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    if not save_flag:
        plt.show()
    else:
        plt.savefig(target_file, dpi=100)


def loadDataH5(h5_file='./data1.h5'):
    with h5py.File(h5_file, 'r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print(trainX.shape, trainY.shape)
        print(valX.shape, valY.shape)

    return trainX, trainY, valX, valY


def load_and_preprocess():
    h5_file = './data1.h5'

    trainX, trainY, testX, testY = loadDataH5(h5_file)

    trainX = tf.keras.utils.normalize(trainX, axis=1)
    testX = tf.keras.utils.normalize(testX, axis=1)

    return trainX, trainY, testX, testY


def get_pre_trained_models():
    # pre-trained models
    vgg16 = tf.keras.applications.VGG16(include_top=False,  # weights='imagenet',
                                        input_shape=(128, 128, 3))
    vgg19 = tf.keras.applications.VGG19(include_top=False,  # weights='imagenet',
                                        input_shape=(128, 128, 3))
    inception3 = tf.keras.applications.InceptionV3(include_top=False,  # weights='imagenet',
                                                   input_shape=(128, 128, 3))

    return vgg16, vgg19, inception3


def get_pre_trained_weights_transformed(pre_trained_model, trainX, testX):
    # predict weights using pre-built models
    pre_trained_trainX = pre_trained_model.predict(trainX)
    pre_trained_testX = pre_trained_model.predict(testX)

    # transform
    pre_trained_trainX = pre_trained_trainX.reshape(pre_trained_trainX.shape[0], -1)
    pre_trained_testX = pre_trained_testX.reshape(pre_trained_testX.shape[0], -1)

    return pre_trained_trainX, pre_trained_testX


def get_model1(pre_trained_model: tf.keras.models.Sequential, model_name, n_classes,
               INCLUDE_DROPOUT=True,
               INCLUDE_CLASSIFICATION=True):
    model = tf.keras.models.Sequential(name=model_name)
    model.add(pre_trained_model)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(150, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))

    INCLUDE_CLASSIFICATION and model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def set_trainable(model, trainable_layers_list):
    model.trainable = True
    trainable_flag = False

    for layer in model.layers:
        if layer.name in trainable_layers_list:
            #             layer.trainable = True
            trainable_flag = True

        layer.trainable = trainable_flag

    return model


def compile_and_fit(model: tf.keras.models.Sequential, trainX, testX, lr=0.01,
                    n_epochs=5,
                    batch_size=32):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=lr),
                  metrics=['accuracy'])

    history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(testX, testY))

    return history


def get_random_forest_classification_accuracy(traindata, testdata, n_estimators, max_depth):
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                 max_depth=max_depth,
                                 n_jobs=-1)
    rfc.fit(traindata, trainY)
    predictions = rfc.predict(testdata)

    accuracy = accuracy_score(testY, predictions)
    print('Random Forest Accuracy: ', accuracy_score(testY, predictions), '\n')
    # print(classification_report(testY, predictions))
    return accuracy


def get_logistic_regression_accuracy(traindata, testdata):
    rfc = sklearn.linear_model.LogisticRegression(random_state=random_seed,
                                                  solver='lbfgs')
    rfc.fit(traindata, trainY)
    predictions = rfc.predict(testdata)

    accuracy = accuracy_score(testY, predictions)
    print('Logistic Regression Accuracy: ', accuracy_score(testY, predictions), '\n')
    # print(classification_report(testY, predictions))
    return accuracy


# def get_logistic_regression_accuracy(traindata, testdata):
#     rfc = sklearn.linear_model.LogisticRegression(random_state=random_seed, n_jobs=-1,
#                                                   solver='sag', max_iter=300)
#     rfc.fit(traindata, trainY)
#     predictions = rfc.predict(testdata)
#
#     accuracy = accuracy_score(testY, predictions)
#     print('Logistic Regression Accuracy: ', accuracy_score(testY, predictions), '\n')
#     # print(classification_report(testY, predictions))
#     return accuracy


def param_cv(feature_train, target_train, feature_test, target_test, estimator, param_grid):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=3, refit=True, n_jobs=3)
    grid_search.fit(feature_train, target_train)

    # print(classifier.best_estimator_)
    print("Best Score: " + str(grid_search.best_score_))
    print("Best Params: " + str(grid_search.best_params_))
    # print("CV Result: " + str(grid_search.cv_results_))

    # refit best params or set refit=True for GridSearchCV
    # grid_search.fit(feature_train, target_train)
    # print("CV Score:" + str(grid_search.score(feature_test, target_test)))


def logistic_regression_param_cv(trainX, testX):
    estimator = sklearn.linear_model.LogisticRegression(n_jobs=-1, random_state=random_seed)

    param_grid = [
        {
            # 'penalty': ['l1', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    ]

    param_cv(trainX, trainY, testX, trainY, estimator=estimator, param_grid=param_grid)


def random_forest_clf_param_cv(trainX, testX):
    estimator = RandomForestClassifier(n_jobs=3, random_state=random_seed)

    param_grid = [
        {
            # 'criterion': ['gini', 'entropy'],
            'n_estimators': list(range(100, 8000, 100)),
            'max_depth': list(range(2, 20, 2))
        }
    ]

    param_cv(trainX, trainY, testX, trainY, estimator=estimator, param_grid=param_grid)


def partB_task1():
    # VGG16
    vgg16_trainX, vgg16_testX = get_pre_trained_weights_transformed(vgg16, trainX, testX)
    print("Model: ", vgg16.name)

    # GridSearch to find best params
    # random_forest_clf_param_cv(vgg16_trainX, vgg16_testX)
    # logistic_regression_param_cv(vgg16_trainX, vgg16_testX)

    # Feed pre-trained weights from base model to Random Forest / Logistic Regression
    get_logistic_regression_accuracy(vgg16_trainX, vgg16_testX)
    get_random_forest_classification_accuracy(vgg16_trainX, vgg16_testX, n_estimators=6000,
                                              max_depth=12)

    # VGG19
    vgg19_trainX, vgg19_testX = get_pre_trained_weights_transformed(vgg19, trainX, testX)
    print("Model: ", vgg19.name)

    # random_forest_clf_param_cv(vgg19_trainX, vgg19_testX)

    # Feed pre-trained weights from base model to Random Forest / Logistic Regression
    get_logistic_regression_accuracy(vgg19_trainX, vgg19_testX)
    get_random_forest_classification_accuracy(vgg19_trainX, vgg19_testX, n_estimators=6000,
                                              max_depth=12)

    # Inception3
    inception3_trainX, inception3_testX = get_pre_trained_weights_transformed(inception3, trainX,
                                                                              testX)
    print("Model: ", inception3.name)

    # random_forest_clf_param_cv(inception3_trainX, inception3_testX)

    # Feed pre-trained weights from base model to Random Forest / Logistic Regression
    get_logistic_regression_accuracy(inception3_trainX, inception3_testX)
    get_random_forest_classification_accuracy(inception3_trainX, inception3_testX,
                                              n_estimators=6000, max_depth=12)


def partB_task2():
    n_classes = 17

    include_dropout = True

    # vgg16_model1 = get_model1(vgg16, 'VGG16', n_classes, INCLUDE_DROPOUT=include_dropout)
    # vgg19_model1 = get_model1(vgg19, 'VGG19', n_classes, INCLUDE_DROPOUT=include_dropout)
    # inception3_model1 = get_model1(inception3, 'Inception3', n_classes,
    #                                INCLUDE_DROPOUT=include_dropout)

    n_epochs = 30
    learning_rate = 0.001

    trainable_layers = ['block1_conv1']

    # set desired layers of the base model as trainable
    vgg16_trainable = set_trainable(vgg16, trainable_layers)

    # add additional Dense layers with Dropout and L2 regularization
    vgg16_trainable_model1 = get_model1(vgg16_trainable, 'vgg16_trainable+', n_classes,
                                        INCLUDE_DROPOUT=True)

    history = compile_and_fit(vgg16_trainable_model1, trainX, testX, lr=learning_rate, n_epochs=n_epochs)
    plot_graph(history, n_epochs)


if __name__ == '__main__':
    random_seed = 171450
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    trainX, trainY, testX, testY = load_and_preprocess()
    vgg16, vgg19, inception3 = get_pre_trained_models()

    partB_task1()
    # TODO uncomment task 2
    # partB_task2()
