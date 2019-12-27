import os
import time
import tensorflow as tf
import tensorflow as tfp
import numpy as np
import pandas as pd
import itertools
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
  
x_test = mnist.test.images
print('Labeled data size :', x_train.shape)
print('Labeled data size :', x_PL.shape)
print('Proportion of class label in train data: ')
pd.DataFrame(np.unique(np.argmax(y_train,1), return_counts = True))

tf.test.is_gpu_available()

# Dropout parameters
dropoutRate_0 = 0.2 #ref
dropoutRate_1 = 0. #ref

# NN parameters
inputN = 784 #default
hiddenN = 5000 #ref 
outputN = 10 #default

# DAE_NN parameters
DAE_hiddenN1 = 256 # ref
DAE_hiddenN2 = 128
DAE_hiddenN3 = 256
destruction_proportion = 0.5 # ref
trainingEpochsDAE = 10
batch_sizeDAE = 256

# iteraction parameters
trainingEpochs = 3000 #ref
batchSize = 32 # ref
PLbatchSize = 256 #ref

# balancing coefficient
T1 = 100 #ref
T2 = 600 #ref
a = 0. #ref
af = 3. #ref

T1_DAE = 200 #ref
T2_DAE = 800 #ref

# SGD with dynamic momentum
learningRate = 1.5 # ref
T = 500 #ref
k = 0.998#ref
pi = 0.5#ref
pf = 0.99#ref

def making_noise(x, prob):
  dim1, dim2 = x.shape
  corrupted_factor = np.concatenate([np.zeros([dim1, int(dim2*prob)]), np.ones([dim1, dim2- int(dim2*prob)])], axis = 1)
  np.apply_along_axis(np.random.shuffle,1,corrupted_factor)
  return x*corrupted_factor

x_train_noise = making_noise(x_train, destruction_proportion)
x_PL_noise = making_noise(x_PL, destruction_proportion)
x_test_noise = making_noise(x_test, destruction_proportion)

x = tf.placeholder("float", [None, inputN])
y = tf.placeholder("float", [None, outputN])
PLx = tf.placeholder("float", [None, inputN])
PLy = tf.placeholder("float", [None, outputN])
DAE_x = tf.placeholder("float", [None, inputN])
DAE_x_noise = tf.placeholder("float", [None, inputN])
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

def DAE_NN(DAE_x_noise, w, b):
  # Hidden layer 1
  HL1 = tf.nn.sigmoid(tf.add(tf.matmul(DAE_x_noise, w['h1']), b['b1']))
  #HL1 = tf.nn.dropout(HL1, rate = dropoutRate_1)
  # Hidden layer 2
  HL2 = tf.nn.sigmoid(tf.add(tf.matmul(HL1, w['h2']), b['b2']))
  #HL2 = tf.nn.dropout(HL2, rate = dropoutRate_1)
  # Hidden layer 3
  HL3 = tf.nn.sigmoid(tf.add(tf.matmul(HL2, w['h3']), b['b3']))
  #HL3 = tf.nn.dropout(HL3, rate = dropoutRate_1)
  # Output layer
  out_layer = tf.nn.sigmoid(tf.add(tf.matmul(HL3, w['out']), b['out']))
  
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

with tf.variable_scope('DAE') :
  weightsDAE = {
    'h1': tf.Variable(tf.random_normal([inputN, DAE_hiddenN1])),
    'h2': tf.Variable(tf.random_normal([DAE_hiddenN1, DAE_hiddenN2])),
    'h3': tf.Variable(tf.random_normal([DAE_hiddenN2, DAE_hiddenN3])),
    'out': tf.Variable(tf.random_normal([DAE_hiddenN3, inputN]))
    }
  biasesDAE = {
    'b1': tf.Variable(tf.random_normal([DAE_hiddenN1])),
    'b2': tf.Variable(tf.random_normal([DAE_hiddenN2])),
    'b3': tf.Variable(tf.random_normal([DAE_hiddenN3])),
    'out': tf.Variable(tf.random_normal([inputN]))
    }

predDAE = DAE_NN(DAE_x_noise, weightsDAE, biasesDAE)

costDAE = tf.reduce_mean(tf.compat.v1.losses.log_loss(DAE_x, predDAE))

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
optimizerDAE = tf.train.AdamOptimizer().minimize(costDAE)                                       

# Initializing the variables
init = tf.global_variables_initializer()

def DAE_preprocessing():
  begin = time.time()
  for epoch in range(50):
    total_loss = 0

    x_all = np.concatenate([x_train, x_PL, x_test], axis = 0)
    x_all_noise = making_noise(x_all, destruction_proportion)
    index = np.arange(x_all.shape[0])
    np.random.shuffle(index)
    total_batch = int(x_all.shape[0] / batch_sizeDAE)

    batches_X = np.array_split(x_all[index], total_batch)
    batches_X_noise = np.array_split(x_all_noise[index], total_batch)
    for i in range(total_batch):
      batch_x, batch_x_noise = batches_X[i], batches_X_noise[i]
      _, l = sess.run([optimizerDAE, costDAE], feed_dict={DAE_x: batch_x,
                                                          DAE_x_noise: batch_x_noise})
      total_loss += l
    print("Epoch {} | time = {:.4f} | MSE loss = {:.4f}".format(epoch, time.time() - begin, total_loss))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
DAE_preprocessing()

x_pred = sess.run(predDAE, feed_dict = {DAE_x: x_train, DAE_x_noise: x_train_noise})
f, a = plt.subplots(3, 10, figsize=(10, 3))
plt.axis('off')
for i in range(10):
    a[0][i].imshow(np.reshape(x_train[i], (28, 28)), cmap='Greys',  interpolation='nearest')
    a[1][i].imshow(np.reshape(x_train_noise[i], (28, 28)), cmap='Greys',  interpolation='nearest')
    a[2][i].imshow(np.reshape(x_pred[i], (28, 28)), cmap='Greys',  interpolation='nearest')
plt.show()
x_PL = sess.run(predDAE, feed_dict = {DAE_x: x_PL, DAE_x_noise: x_PL_noise})
x_test = sess.run(predDAE, feed_dict = {DAE_x: x_test, DAE_x_noise: x_test_noise})
x_train = x_pred
sess.close()

# Launch the graph
def accuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predNN, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: x_test, y: mnist.test.labels})


def accuracytestPL():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predPL, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: x_test, y: mnist.test.labels})

def dropNN_PL():
  ### Neural Network parameters
  iteration_list = []
  neural_network_accuracy_list = []
  pseudo_label_accuarcy_list = []
  neural_network_accuracy = 0
  pseudo_label_accuarcy = 0
  iteration = 0
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
            print("Epoch {} | time = {:.4f} | DropNN+DAE acc = {:.4f} | DropNN+PL+DAE acc = {:.4f} "
                  .format(epoch, time.time() - begin,
                          neural_network_accuracy, pseudo_label_accuarcy))

            iteration_list = np.append(iteration_list, iteration)
            neural_network_accuracy_list = np.append(neural_network_accuracy_list, neural_network_accuracy)
            pseudo_label_accuarcy_list = np.append(pseudo_label_accuarcy_list, pseudo_label_accuarcy)

        plt.plot(iteration_list, pseudo_label_accuarcy_list, iteration_list, neural_network_accuracy_list, 'r--')
    print("Optimization Finished!")
    print("DropoutNN+DAE:", accuracytestNN())
    print("+PL+DAE:", accuracytestPL())
    plt.show()

dropNN_PL()
