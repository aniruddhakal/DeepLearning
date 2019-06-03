import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time


def preprocess_train_test_data(train_images, train_labels, test_images, test_labels,
                               classes_included):
    # condition to select images only with classes from classes_included array
    train_condition = np.isin(train_labels, classes_included)
    test_condition = np.isin(test_labels, classes_included)

    # select only the labels which satisfy the above conditions
    train_labels = train_labels[train_condition]
    test_labels = test_labels[test_condition]

    # convert first class's labels to 0
    train_labels[train_labels == classes_included[0]] = 0
    test_labels[test_labels == classes_included[0]] = 0

    # convert second class's labels to 1
    train_labels[train_labels == classes_included[1]] = 1
    test_labels[test_labels == classes_included[1]] = 1

    # transpose and convert rank-0 label arrays to rank-1 label arrays
    train_labels = train_labels.T.reshape(1, -1)
    test_labels = test_labels.T.reshape(1, -1)

    # select training images with only only two classes
    train_images = train_images[train_condition]

    # select test images with only only two classes
    test_images = test_images[test_condition]

    # flatten the images
    train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float32)

    # normalize the train and test data
    train_images = (train_images / 255.0).T
    test_images = (test_images / 255.0).T

    return train_images, train_labels, test_images, test_labels


def task1():
    n_epochs = 50
    n_hidden = 100
    n_outputs = 1
    learning_rate = 0.01

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # considered classes of image for classification
    classes_included = [3, 8]

    # preprocess the data
    train_images, train_labels, test_images, test_labels = preprocess_train_test_data(train_images,
                                                                                      train_labels,
                                                                                      test_images,
                                                                                      test_labels,
                                                                                      classes_included)

    n_inputs = train_images.shape[0]
    print("n_inputs: ", n_inputs)

    print("Data Extracted and Reshaped")
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    # reset default graph
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [n_inputs, None], name='image')
    Y = tf.placeholder(tf.float32, [n_outputs, None], name='class')

    # weight and bias matrices
    W1 = tf.get_variable('w1', shape=(n_hidden, n_inputs))
    B1 = tf.get_variable('b1', shape=(n_hidden, 1), initializer=tf.zeros_initializer())

    W2 = tf.get_variable('w2', shape=(n_outputs, n_hidden))
    B2 = tf.get_variable('b2', shape=(n_outputs, 1), initializer=tf.zeros_initializer())

    # push through the hidden layer
    A1 = tf.add(tf.matmul(W1, X), B1)
    h1 = tf.nn.relu(A1)

    # push the output of hidden layer through the final/output layer
    A2 = tf.add(tf.matmul(W2, h1), B2)
    predicted = tf.sigmoid(A2)

    # calculate the error for using sigmoid loss function
    error = tf.nn.sigmoid_cross_entropy_with_logits(logits=A2, labels=Y)
    loss = tf.reduce_mean(error)

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predicted_round = tf.round(predicted)

    # calculate correct predictions
    correct_predictions = tf.cast(tf.equal(predicted_round, Y), tf.float32)

    # calculate accuracy on the test set
    accuracy = tf.reduce_mean(correct_predictions)

    with tf.Session() as sess:
        # a dictionary to save information for all epochs
        epoch_history = {
            'epoch': list(),
            'loss': list(),
            'training_accuracy': list(),
            'validation_accuracy': list(),
            'validation_loss': list()
        }

        sess.run(tf.global_variables_initializer())

        start_time = time()
        for epoch in range(n_epochs):
            _, current_loss, acc = sess.run([optimizer, loss, accuracy],
                                            feed_dict={X: train_images, Y: train_labels})

            print("Iteration", epoch, "Loss: ", current_loss, " Training Accuracy: ", acc)
            validation_accuracy, validation_loss = sess.run([accuracy, loss],
                                                            feed_dict={X: test_images,
                                                                       Y: test_labels})
            print("Final validation_accuracy: ", validation_accuracy)
            epoch_history = update_epoch_history(epoch, current_loss, acc, validation_accuracy,
                                                 validation_loss, epoch_history)

        finish_time = time() - start_time
        plot_training_graph(epoch_history)

        print("Test Accuracy: ", sess.run([accuracy], feed_dict={X: test_images, Y: test_labels}))
        print("Training Time: %f seconds" % finish_time)


def preprocess_train_test_data2(train_images, train_labels, test_images, test_labels):
    # number of unique labels
    num_labels = np.unique(train_labels).size

    # conversion to categorical labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_labels)

    train_labels = train_labels.T
    test_labels = test_labels.T

    # flatten the images
    train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float32)

    # normalize the train and test data
    train_images = (train_images / 255.0).T
    test_images = (test_images / 255.0).T

    return train_images, train_labels, test_images, test_labels


def task2(USE_MINIBATCH=True, mini_batch_size=512):
    n_epochs = 40
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    learning_rate = 0.01

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # preprocess the data
    train_images, train_labels, test_images, test_labels = preprocess_train_test_data2(train_images,
                                                                                       train_labels,
                                                                                       test_images,
                                                                                       test_labels)

    n_inputs = train_images.shape[0]
    print("n_inputs: ", n_inputs)

    print("Data Extracted and Reshaped")
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    # reset default graph
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [n_inputs, None], name='image')
    Y = tf.placeholder(tf.float32, [n_outputs, None], name='class')

    # weight and bias matrices
    W1 = tf.get_variable('w1', shape=(n_hidden1, n_inputs))
    B1 = tf.get_variable('b1', shape=(n_hidden1, 1), initializer=tf.zeros_initializer())

    W2 = tf.get_variable('w2', shape=(n_hidden2, n_hidden1))
    B2 = tf.get_variable('b2', shape=(n_hidden2, 1), initializer=tf.zeros_initializer())

    W3 = tf.get_variable('w3', shape=(n_outputs, n_hidden2))
    B3 = tf.get_variable('b3', shape=(n_outputs, 1), initializer=tf.zeros_initializer())

    # push through the hidden layer
    A1 = tf.add(tf.matmul(W1, X), B1)
    h1 = tf.nn.relu(A1)

    # push the output of hidden layer through the final/output layer
    A2 = tf.add(tf.matmul(W2, h1), B2)
    h2 = tf.nn.relu(A2)

    A3 = tf.add(tf.matmul(W3, h2), B3)

    logits = tf.transpose(A3)
    labels = tf.transpose(Y)

    # calculate the error for using sigmoid loss function
    error = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    loss = tf.reduce_mean(error)

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # calculate correct predictions
    correct_predictions = tf.cast(tf.equal(tf.argmax(A3), tf.argmax(Y)), tf.float32)

    # calculate accuracy on the test set
    accuracy = tf.reduce_mean(correct_predictions)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # a dictionary to save information for all epochs
        epoch_history = {
            'epoch': list(),
            'loss': list(),
            'training_accuracy': list(),
            'validation_accuracy': list(),
            'validation_loss': list()
        }

        start_time = time()
        for epoch in range(n_epochs):
            start = 0

            if USE_MINIBATCH:
                while start < train_images.shape[1]:
                    # mini-batch update
                    # select mini_batch_size number of indices
                    train_images_batch = train_images[:, start:(start + mini_batch_size)]
                    train_labels_batch = train_labels[:, start:(start + mini_batch_size)]

                    # TODO add validation accuracy
                    # update index to choose next mini_batch_size samples for next batch
                    start += mini_batch_size
                    _, current_loss, acc = sess.run([optimizer, loss, accuracy],
                                                    feed_dict={X: train_images_batch,
                                                               Y: train_labels_batch})
                print("Epoch: ", (epoch + 1), "Loss: ", current_loss, " Training Accuracy: ", acc)
                validation_accuracy, validation_loss = sess.run([accuracy, loss],
                                                                feed_dict={X: test_images,
                                                                           Y: test_labels})
                print("Final validation_accuracy: ", validation_accuracy)
                epoch_history = update_epoch_history(epoch, current_loss, acc, validation_accuracy,
                                                     validation_loss, epoch_history)
            else:
                _, current_loss, acc = sess.run([optimizer, loss, accuracy],
                                                feed_dict={X: train_images, Y: train_labels})

                print("Epoch: ", (epoch + 1), "Loss: ", current_loss, " Training Accuracy: ", acc)
                validation_accuracy, validation_loss = sess.run([accuracy, loss],
                                                                feed_dict={X: test_images,
                                                                           Y: test_labels})
                print("Final validation_accuracy: ", validation_accuracy)
                epoch_history = update_epoch_history(epoch, current_loss, acc, validation_accuracy,
                                                     validation_loss, epoch_history)

        finish_time = time() - start_time
        if USE_MINIBATCH:
            print("Training Time: %f seconds" % finish_time, "Batch_Size: %d" % mini_batch_size)
        else:
            print("Training Time: %f seconds" % finish_time, "No Mini-Batch")

        plot_training_graph(epoch_history)

        return finish_time


def update_epoch_history(epoch, current_loss, acc, validation_accuracy, validation_loss,
                         epoch_history):
    """
    append details of each epoch to appropriate lists in history dictionary
    :param epoch:
    :param current_loss:
    :param acc:
    :param epoch_history:
    :return:
    """
    epoch_list = epoch_history['epoch']
    loss_list = epoch_history['loss']
    training_accuracy_list = epoch_history['training_accuracy']
    validation_accuracy_list = epoch_history['validation_accuracy']
    validation_loss_list = epoch_history['validation_loss']

    epoch_list.append(epoch)
    loss_list.append(current_loss)
    training_accuracy_list.append(acc)
    validation_accuracy_list.append(validation_accuracy)
    validation_loss_list.append(validation_loss)

    epoch_history['epoch'] = epoch_list
    epoch_history['loss'] = loss_list
    epoch_history['training_accuracy'] = training_accuracy_list
    epoch_history['validation_accuracy'] = validation_accuracy_list
    epoch_history['validation_loss'] = validation_loss_list

    return epoch_history


def plot_training_graph(epoch_history):
    """
    this plot code is copied from Week 9 lecture slides of 'Deep Learning'
    """
    n_epochs = len(epoch_history['epoch'])

    plt.style.use("ggplot")
    plt.figure(dpi=300)
    plt.plot(np.arange(0, n_epochs), epoch_history['loss'], label="train_loss")
    plt.plot(np.arange(0, n_epochs), epoch_history["validation_loss"], label="Validation_loss")
    plt.plot(np.arange(0, n_epochs), epoch_history['training_accuracy'], label="train_acc")
    plt.plot(np.arange(0, n_epochs), epoch_history["validation_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def plot_final_graph(run_history):
    n_runs = len(run_history['test_acc'])

    # plt.style.use("ggplot")
    # plt.figure(dpi=300)
    # # plt.plot(np.arange(0, n_runs), run_history['test_acc'], label="Test Accuracy")
    # plt.plot(run_history["finish_time"], label="Finish Time")
    # plt.xticks(run_history['batch_size'])
    # plt.yticks(run_history['finish_time'])
    # plt.title("Finish Time and Test Accuracy")
    # plt.xlabel("Run #")
    # plt.ylabel("Accuracy")
    # plt.legend()
    plt.clf()
    plt.plot(run_history['finish_time'])
    plt.xticks(run_history['batch_size'])
    # plt.yticks(run_history['finish_time'])
    plt.xlabel("Batch size")
    plt.ylabel("Finish Time")
    plt.show()


def update_run_history(run_history, batch_size, test_acc, finish_time):
    test_acc_list = run_history['test_acc']
    finish_time_list = run_history['finish_time']
    batch_size_list = run_history['batch_size']

    test_acc_list.append(test_acc)
    finish_time_list.append(finish_time)
    batch_size_list.append(batch_size)

    run_history['test_acc'] = test_acc_list
    run_history['finish_time'] = finish_time_list
    run_history['batch_size'] = batch_size_list

    return run_history


if __name__ == '__main__':
    run_history = {
        'test_acc': list(),
        'finish_time': list(),
        'batch_size': list()
    }

    # Part A - (i)
    # task1()

    # Part A - (ii)
    # finish_time = task2(USE_MINIBATCH=False)
    # print("Time taken %f seconds" % finish_time)

    # Part A - (iii)
    for mini_batch_size in [32, 64, 128, 256, 512, 1024, 2048]:
        finish_time = task2(USE_MINIBATCH=True, mini_batch_size=mini_batch_size)

    # print("Time taken %f seconds" % finish_time)
