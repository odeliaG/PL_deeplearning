import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import StratifiedShuffleSplit

mnist = input_data.read_data_sets("", one_hot=True)

stratSplit = StratifiedShuffleSplit(test_size=100, random_state=2812, n_splits=1)
stratSplit.get_n_splits(mnist.train.images, np.argmax(mnist.train.labels, axis = 1))
for train_index, test_index in stratSplit.split(X = mnist.train.images,y = mnist.train.labels):
  x_train = mnist.train.images[test_index]
  y_train = mnist.train.labels[test_index]
  x_PL = mnist.train.images[train_index]

print('Labeled data size :', x_train.shape)
print('Labeled data size :', x_PL.shape)
print('Proportion of class label in train data: ')
pd.DataFrame(np.unique(np.argmax(y_train,1), return_counts = True))

tf.test.is_gpu_available()

# Neural Network parameters
iteration_list = []
neural_network_accuracy_list = []
pseudo_label_accuarcy_list = []
neural_network_accuracy = 0
pseudo_label_accuarcy = 0

learningRate = 1.5
trainingEpochs = 3000
dropoutRate_0 = 0.2
dropoutRate_1 = 0.5

inputN = 784
hiddenN = 5000
outputN = 10

batchSize = 32
PLbatchSize = 256

iteration = 0
cPL = 0

T1 = 100
T2 = 600
a = 0.
af = 3.

T = 500
k = 0.998
pi = 0.5
pf = 0.99

x = tf.placeholder("float", [None, inputN])
y = tf.placeholder("float", [None, outputN])
PLx = tf.placeholder("float", [None, inputN])
PLy = tf.placeholder("float", [None, outputN])
alpha_t = tf.placeholder("float", )
p_t = tf.placeholder("float",)
epsilon_t = tf.placeholder("float",)
plt.clf()

def NN(x, w, b):
    # Hidden layer 1
    HL = tf.add(tf.matmul(x, w['h1']), b['b1'])
    HL = tf.nn.relu(HL)
    HL = tf.nn.dropout(HL, rate = dropoutRate_1)
    # Output layer
    out_layer =tf.matmul(HL, w['out']) + b['out']

    return out_layer

# initialize weights and biases
with tf.variable_scope('NN') :
  weightsNN = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'out': tf.Variable(tf.random_normal([hiddenN, outputN]))
    }

  biasesNN = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'out': tf.Variable(tf.random_normal([outputN]))
    }

with tf.variable_scope('PL') :
  weightsPL = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'out': tf.Variable(tf.random_normal([hiddenN, outputN]))
    }
  biasesPL = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'out': tf.Variable(tf.random_normal([outputN]))
    }

predNN = NN(x, weightsNN, biasesNN)
predPL = NN(x, weightsPL, biasesPL)
predPL1 = NN(PLx, weightsPL, biasesPL)

costNN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predNN,
                                                                labels=y))

costPL = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predPL,
                                                                       labels=y)),
                (alpha_t * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predPL1,
                                                                                labels=PLy))))

# Gradient Descent
optimizerNN = tf.train.MomentumOptimizer(learning_rate = (1-p_t)*epsilon_t,
                                        momentum = -p_t/(1-p_t)).minimize(costNN)
optimizerPL = tf.train.MomentumOptimizer(learning_rate = (1-p_t)*epsilon_t,
                                        momentum = -p_t/(1-p_t)).minimize(costNN)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
def accuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predNN, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


def accuracytestPL():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predPL, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


with tf.Session() as sess:
    sess.run(init)
    begin=time.time()
    # Training cycle
    for epoch in range(trainingEpochs):
        avg_costNN = 0.
        avg_costPL = 0.
        total_batch = int(100 / batchSize)
        # Loop over all batches
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        batches_X = np.array_split(x_train[index], total_batch)
        batches_y = np.array_split(y_train[index], total_batch)
        index = np.arange(x_PL.shape[0])
        np.random.shuffle(index)
        batches_X_PL = np.array_split(x_PL[index], total_batch)
        for i in range(total_batch):
            batch_x, batch_y = batches_X[i], batches_y[i]
            if epoch > T:
              p = pf
            else:
              p = epoch/T*pf + (1-epoch/T)*pi

            _, cNN = sess.run([optimizerNN, costNN], feed_dict={x: batch_x,
                                                                y: batch_y,
                                                                p_t : p,
                                                                epsilon_t : learningRate*(k**epoch)
                                                                })
            if epoch < T1:
                a = 0
            elif epoch < T2:
                a = ((epoch - T1) / (T2 - T1)) * af
            else :
                a = af
            

            # Pseudolabel
            batch_xpred = batches_X_PL[i]
            batch_ypred = sess.run([predPL], feed_dict={x: batch_xpred})
            batch_ypred = batch_ypred[0]
            batch_ypred = batch_ypred.argmax(1)
            batch_ypre = np.zeros((batch_xpred.shape[0], 10))
            for ii in range(PLbatchSize):
                batch_ypre[ii, batch_ypred[ii]] = 1

            _, cPL = sess.run([optimizerPL, costPL], feed_dict={x: batch_x,
                                                                y: batch_y,
                                                                PLx: batch_xpred,
                                                                PLy: batch_ypre,
                                                                p_t : p,
                                                                epsilon_t : learningRate*(k**epoch),
                                                                alpha_t: a})
            iteration += 1
            # Compute average loss
            avg_costNN += cNN / total_batch
            avg_costPL += cPL / total_batch

        if epoch % 100 == 0:
            neural_network_accuracy = accuracytestNN()
            pseudo_label_accuarcy = accuracytestPL()
            print("Epoch {} | time = {:.4f} | DropNN acc = {:.4f} | DropNN+PL acc = {:.4f} "
                  .format(epoch, time.time() - begin,
                          neural_network_accuracy, pseudo_label_accuarcy))

            iteration_list = np.append(iteration_list, iteration)
            neural_network_accuracy_list = np.append(neural_network_accuracy_list, neural_network_accuracy)
            pseudo_label_accuarcy_list = np.append(pseudo_label_accuarcy_list, pseudo_label_accuarcy)

        plt.plot(iteration_list, pseudo_label_accuarcy_list, iteration_list, neural_network_accuracy_list, 'r--')
    print("Optimization Finished!")
    print("Neural Network accuracy:", accuracytestNN())
    print("+PL:", accuracytestPL())
    plt.show()
