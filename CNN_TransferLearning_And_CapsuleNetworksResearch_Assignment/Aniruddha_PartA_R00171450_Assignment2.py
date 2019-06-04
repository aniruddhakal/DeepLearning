import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

import h5py


def plot_graph(history, n_epochs, save_flag, target_file):
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

    if not save_flag:
        plt.show()
    else:
        plt.savefig(target_file, dpi=300)


def loadDataH5(h5_file='./data1.h5'):
    with h5py.File(h5_file, 'r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print(trainX.shape, trainY.shape)
        print(valX.shape, valY.shape)

    return trainX, trainY, valX, valY


def get_default_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filter_counts[0], (5, 5), input_shape=input_shape,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(rate=0.2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def get_config1_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filter_counts[0], (5, 5), input_shape=input_shape,
                                     activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.2))

    model.add(tf.keras.layers.Conv2D(filter_counts[1], (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.15))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def get_config2_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filter_counts[0], (5, 5), input_shape=input_shape,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.2))

    model.add(tf.keras.layers.Conv2D(filter_counts[1], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.15))

    model.add(tf.keras.layers.Conv2D(filter_counts[2], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.07))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.07))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def get_config3_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filter_counts[0], (5, 5), input_shape=input_shape,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.2))

    model.add(tf.keras.layers.Conv2D(filter_counts[1], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.15))

    model.add(tf.keras.layers.Conv2D(filter_counts[2], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.1))

    model.add(tf.keras.layers.Conv2D(filter_counts[3], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.SpatialDropout2D(0.07))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.07))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    INCLUDE_DROPOUT and model.add(tf.keras.layers.Dropout(0.03))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def get_dynamic_diverse_data_generators():
    zoom_range = np.random.choice([-0.2, -0.3, -0.1, 0.2])
    shear_range = np.random.choice([0.2, 0.3, 0.1])
    rotation_range = np.random.choice([0, 0.5, 0.4, 0.2, -0.2, -0.5, -0.4])
    horizontal_flip = np.random.choice([True, False])
    vertical_flip = np.random.choice([True, False])
    featurewise_normalization = np.random.choice([True, False])
    samplewise_normalization = np.random.choice([True, False])

    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_std_normalization=featurewise_normalization,
        samplewise_std_normalization=samplewise_normalization,
        zoom_range=zoom_range,
        shear_range=shear_range,
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    # validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_std_normalization=featurewise_normalization,
    #     samplewise_std_normalization=samplewise_normalization,
    #     zoom_range=zoom_range,
    #     shear_range=shear_range,
    #     rotation_range=rotation_range,
    #     horizontal_flip=horizontal_flip,
    #     vertical_flip=vertical_flip
    # )

    train_generator = train_data_generator.flow(trainX, trainY, seed=random_seed)
    # validation_generator = validation_data_generator.flow(testX, testY, seed=random_seed)

    return train_generator


def get_data_generators():
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_std_normalization=True, samplewise_std_normalization=True,
        zoom_range=-0.2,
        shear_range=0.2,
        # rotation_range=0.5,
        horizontal_flip=True,
        vertical_flip=False)

    validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_std_normalization=True, samplewise_std_normalization=True,
        zoom_range=-0.2,
        shear_range=0.2,
        # rotation_range=0.5,
        horizontal_flip=True,
        vertical_flip=False
    )

    train_generator = train_data_generator.flow(trainX, trainY, seed=random_seed)
    validation_generator = validation_data_generator.flow(testX, testY, seed=random_seed)

    return train_generator, validation_generator


def main(trainX, trainY, testX, testY, configuration, save_flag, target_file, filter_counts,
         INCLUDE_DROPOUT, USE_AUGMENTATION):
    learning_rate = 0.01
    n_epochs = 70
    batch_size = 128

    input_shape = trainX.shape[1:]

    # number of unique classes, used to create N softmax layers
    n_classes = np.unique(trainY).size

    if configuration is 'config1':
        model = get_config1_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    elif configuration is 'config2':
        model = get_config2_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    elif configuration is 'config3':
        model = get_config3_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    else:
        # baseline model
        model = get_default_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    # optimizer = tf.keras.optimizers.Adadelta(lr=learning_rate)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer, metrics=['accuracy'])

    if USE_AUGMENTATION:
        # augmentation fit-generator
        train_generator, validation_generator = get_data_generators()

        train_steps = trainX.shape[0] // batch_size
        validation_steps = testX.shape[0] // batch_size

        history = model.fit_generator(train_generator, steps_per_epoch=train_steps,
                                      epochs=n_epochs,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps)

        plot_graph(history, n_epochs, save_flag, target_file)
    else:
        history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size,
                            epochs=n_epochs)

        plot_graph(history, n_epochs, save_flag, target_file)


def ensemble_run(trainX, trainY, testX, testY, filter_counts, INCLUDE_DROPOUT, n_epochs):
    learning_rate = 0.01
    n_epochs = n_epochs
    batch_size = 64

    # number of unique classes, used to create N softmax layers
    n_classes = np.unique(trainY).size

    input_shape = trainX.shape[1:]

    filter_counts2 = [20, 40, 80, 160]

    # Diversity through different models and filter counts in each layers
    model1 = get_default_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    model2 = get_config1_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    model3 = get_config2_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    model4 = get_config3_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
    model5 = get_default_model_part1(input_shape, n_classes, filter_counts2, INCLUDE_DROPOUT)
    model6 = get_config1_model_part1(input_shape, n_classes, filter_counts2, INCLUDE_DROPOUT)
    model7 = get_config2_model_part1(input_shape, n_classes, filter_counts2, INCLUDE_DROPOUT)
    model8 = get_config3_model_part1(input_shape, n_classes, filter_counts2, INCLUDE_DROPOUT)

    models = [model1, model2, model3, model4, model5, model6, model7, model8]
    #     models = [model1]

    train_steps = trainX.shape[0] // batch_size
    # validation_steps = testX.shape[0] // batch_size

    results_dict = dict()
    accuracies_dict = dict()

    for model in models:
        # Diversity in augmentation through dynamic setting changes in data generators
        train_generator = get_dynamic_diverse_data_generators()

        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
                      metrics=['accuracy'])

        history = model.fit_generator(train_generator, steps_per_epoch=train_steps,
                                      epochs=n_epochs)
        results = model.predict(testX)

        results_dict[str(model)] = results
        predictions = np.argmax(results, axis=1)
        accuracy = accuracy_score(testY, predictions)
        accuracies_dict[str(model)] = accuracy

    return results_dict, accuracies_dict


# def ensemble(trainX, trainY, testX, testY, save_flag, target_file,
#              filter_counts, INCLUDE_DROPOUT, USE_AUGMENTATION, n_epochs):
#     learning_rate = 0.01
#     n_epochs = n_epochs
#     batch_size = 64
#
#     # number of unique classes, used to create N softmax layers
#     n_classes = np.unique(trainY).size
#
#     input_shape = trainX.shape[1:]
#
#     model1 = get_default_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
#     model2 = get_config1_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
#     model3 = get_config2_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
#     model4 = get_config3_model_part1(input_shape, n_classes, filter_counts, INCLUDE_DROPOUT)
#
#     models = [model1, model2, model3, model4]
#     # models = [model1]
#
#     train_generator, validation_generator = get_data_generators()
#
#     train_steps = trainX.shape[0] // batch_size
#     validation_steps = testX.shape[0] // batch_size
#
#     results_dict = dict()
#     accuracies_dict = dict()
#
#     results = None
#     for model in models:
#         model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
#                       optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
#                       metrics=['accuracy'])
#
#         history = model.fit_generator(train_generator, steps_per_epoch=train_steps,
#                                       epochs=n_epochs)
#         #                                       validation_data=validation_generator,
#         #                                       validation_steps=validation_steps)
#         # results = model.predict_generator(validation_generator,
#         #                                   steps=validation_steps)
#         results = model.predict(testX)
#
#         results_dict[str(model)] = results
#         predictions = np.argmax(results, axis=1)
#         accuracy = accuracy_score(testY, predictions)
#         accuracies_dict[str(model)] = accuracy
#
#     return results_dict, accuracies_dict


def get_average_ensemble_accuracy(accuracies_dict, result_dict, weighted_average=True):
    all_results = np.array(list(results_dict.values()))

    if weighted_average:
        accuracy_weights = list(accuracies_dict.values())
        final_result = np.average(all_results, axis=0, weights=np.array(accuracy_weights))
    else:
        final_result = np.average(all_results, axis=0)

    final_predictions = final_result.argmax(axis=1)

    print(classification_report(testY, final_predictions))

    accuracy = accuracy_score(testY, final_predictions) * 100

    return accuracy


def load_and_preprocess():
    h5_file = './data1.h5'

    trainX, trainY, testX, testY = loadDataH5(h5_file)

    trainX = tf.keras.utils.normalize(trainX, axis=1)
    testX = tf.keras.utils.normalize(testX, axis=1)

    return trainX, trainY, testX, testY


if __name__ == '__main__':
    # random seed
    random_seed = 171450
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)

    INCLUDE_DROPOUT = True
    USE_AUGMENTATION = True
    USE_REGULARIZATION = True
    save_flag = False

    trainX, trainY, testX, testY = load_and_preprocess()

    layers_filter_counts = [20, 128, 192, 256]
    filters = '_64_128_192_256_filters_'

    if INCLUDE_DROPOUT:
        filters += 'dropout_'

    if USE_AUGMENTATION:
        filters += '_best_augmentation_'

    if USE_REGULARIZATION:
        filters += '_regularization_'

    # custom models, default is the baseline model
    configs = ['default', 'config1', 'config2', 'config3']
    # configs = ['default']

    # TODO uncomment
    for configuration in configs:
        target_file = './data/_' + configuration + '_plot_' + filters + '_deep_dense_'
        # target_file = "./data/neagtive_zoom_with_shear_vertical_flip_"
        main(trainX, trainY, testX, testY, configuration, save_flag, target_file,
             layers_filter_counts, INCLUDE_DROPOUT, USE_AUGMENTATION)

    # this below code is only for Ensemble
    n_epochs = 70
    # run ensemble models for n_epochs
    results_dict, accuracy_dict = ensemble_run(trainX, trainY, testX, testY,
                                               layers_filter_counts, INCLUDE_DROPOUT, n_epochs)

    # Calculate Average Ensemble Accuracy - weighted average
    weighted_average_flag = False
    accuracy = get_average_ensemble_accuracy(accuracy_dict, results_dict, weighted_average_flag)
    print('Ensemble Accuracy: ', accuracy)
