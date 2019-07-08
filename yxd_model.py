from __future__ import print_function
import sklearn as sk
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib import rnn
from sklearn.model_selection import KFold, cross_val_score
import csv
from sklearn.utils import shuffle
import os
from read_data_modify import read_data, return_labels

# 添加GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import WiFi Activity data
# csv_convert(window_size,threshold)
# from cross_vali_input_data import csv_import, DataSet

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)  # _num_examples个元素的array
            np.random.shuffle(perm)  # 打乱perm的顺序
            self._images = self._images[perm]  # 将images的行，按照perm的顺序重新排序生成新的矩阵
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


window_size = 600
threshold = 60

# Parameters
learning_rate = 0.0001
batch_size = 16
display_step = 50

# Network Parameters
n_input = 90  # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size  # timesteps
n_hidden = 200  # hidden layer num of features original 200

print('正在训练-- WiFi_data_old_20 --数据集')
train_data_dir, activity_list = return_labels('WiFi_data_old_20')  # 返回data_name数据集所在的文件目录和数据集的标签

n_classes = len(activity_list)  # WiFi activity total classes

# Output folder
OUTPUT_FOLDER_PATTERN = "compare/LR{0}_BATCHSIZE{1}_NHIDDEN{2}/"
output_folder = OUTPUT_FOLDER_PATTERN.format(learning_rate, batch_size, n_hidden)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
mask_x = tf.placeholder(tf.float32, [None, n_hidden])
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    # lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden)  # GRU cell

    # Get lstm cell output
    outputs, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)
    outputs = tf.reduce_sum(outputs, 0) / mask_x
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights['out']) + biases['out']


##### main #####
pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print('activity_list: ', activity_list)
x_train, y_train, x_test, y_test = read_data(train_data_dir, activity_list)   #  xx = xx.reshape(len(xx),1000,90)
print(x_train.shape)
x_train = np.array(x_train).transpose((0, 2, 1, 3)).reshape((len(x_train), n_steps, -1))
print(x_train.shape)
x_test = np.array(x_test).transpose((0, 2, 1, 3)).reshape((len(x_test), n_steps, -1))

cvscores = []
confusion_sum = [[0 for i in range(n_classes)] for j in range(n_classes)]

# k_fold
kk = 10

# Launch the graph
with tf.Session() as sess:
    for i in range(kk):

        # Initialization
        train_loss = []
        train_acc = []
        validation_loss = []
        validation_acc = []

        # data set
        wifi_train = DataSet(x_train, y_train)
        wifi_validation = DataSet(x_test, y_test)

        saver = tf.train.Saver()
        sess.run(init)
        step = 1
        training_iters = 3000
        # Keep training until reach max iterations
        while step < training_iters:
            batch_x, batch_y = wifi_train.next_batch(batch_size)
            x_vali = wifi_validation.images[:]
            y_vali = wifi_validation.labels[:]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            x_vali = x_vali.reshape((-1, n_steps, n_input))
            # Run optimization op (backprop)
            group = batch_size

            sum_num = np.ones([group, n_hidden], float)
            e = np.ones([group, n_hidden], float)
            for v in range(n_steps - 1):
                sum_num = e + sum_num
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, mask_x: sum_num})

            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, mask_x: sum_num})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, mask_x: sum_num})
            # Calculate batch loss
            group = len(x_vali)

            sum_num = np.ones([group, n_hidden], float)
            e = np.ones([group, n_hidden], float)
            for v in range(n_steps - 1):
                sum_num = e + sum_num
            acc_vali = sess.run(accuracy, feed_dict={x: x_vali, y: y_vali, mask_x: sum_num})
            loss_vali = sess.run(cost, feed_dict={x: x_vali, y: y_vali, mask_x: sum_num})

            # Store the accuracy and loss
            train_acc.append(acc)
            train_loss.append(loss)
            validation_acc.append(acc_vali)
            validation_loss.append(loss_vali)

            if step % display_step == 0:
                print("Iter " + str(step) + ", Minibatch Training  Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", Minibatch Validation  Loss= " + \
                      "{:.6f}".format(loss_vali) + ", Validation Accuracy= " + \
                      "{:.5f}".format(acc_vali))
            step += 1

        # Calculate the confusion_matrix
        cvscores.append(acc_vali * 100)
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_vali, y: y_vali, mask_x: sum_num})
        y_true = np.argmax(y_vali, 1)
        print(sk.metrics.confusion_matrix(y_true, y_pred))
        confusion = sk.metrics.confusion_matrix(y_true, y_pred)
        confusion_sum = confusion_sum + confusion

        # Save the Accuracy curve
        fig = plt.figure(2 * i - 1)
        plt.plot(train_acc)
        plt.plot(validation_acc)
        plt.xlabel("n_epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train_acc", "validation_acc"], loc=4)
        plt.ylim([0, 1])
        plt.savefig((output_folder + "Accuracy_" + str(i) + ".png"), dpi=150)

        # Save the Loss curve
        fig = plt.figure(2 * i)
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.xlabel("n_epoch")
        plt.ylabel("Loss")
        plt.legend(["train_loss", "validation_loss"], loc=1)
        plt.ylim([0, 2])
        plt.savefig((output_folder + "Loss_" + str(i) + ".png"), dpi=150)

    print("Optimization Finished!")
    print("%.1f%% (+/- %.1f%%)" % (np.mean(cvscores), np.std(cvscores)))
    saver.save(sess, output_folder + "model.ckpt")

    # Save the confusion_matrix
    np.savetxt(output_folder + "confusion_matrix.txt", confusion_sum, delimiter=",", fmt='%d')
    np.savetxt(output_folder + "accuracy.txt", (np.mean(cvscores), np.std(cvscores)), delimiter=".", fmt='%.1f')
