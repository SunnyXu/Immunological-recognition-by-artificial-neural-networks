#!/usr/bin/env python2.7


# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to filter out warning set as 2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy import *

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


#type 0: fully connected
def init_weights0(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape,stddev=0.1)
    return tf.Variable(weights)


#type 1: hidden layer = 75 #
def init_weights(shape):
    """ Weight initialization """
    sess=tf.Session()
    A = tf.random_normal(((shape[0]-shape[1]),shape[1]),stddev=0.1)
    K = tf.random_normal((1,shape[1]),stddev=0.1)
    K0= sess.run(K)
    B = tf.diag(K0[0])
    weights = tf.concat([A,B],0)  # concatenate tensors along one dimension, 0: along the column
    return tf.Variable(weights)



#type 1: hidden layer = 75 #

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
#temporarily set weights matrix for additional edition later
    W=tf.constant(sess.run(w_1))
    input_layer=sess.run(W).shape[0] # node # of input layer
    hidden_layer=sess.run(W).shape[1] # node # of hidden layer
    diff=input_layer-hidden_layer

#setup the diagnal matrix D and do the multiplication
#to setup the initial h_elem
    for k in range (diff,(diff+1)):
        O=tf.constant([[1.]]) # first element for bias
        for i in range (1,diff):
            elem=tf.constant([[1.]])
            O=tf.concat([O,elem],1)
        for j in range (diff,input_layer):
            if j==k:
                elem=tf.constant([[1.]])
            else:
                elem=tf.constant([[0.]])
            O=tf.concat([O,elem],1)
        D=tf.diag(O[0])
        h_elem=tf.matmul(X,D)
        h_elem=tf.matmul(h_elem,W[:,(k-diff):(k-diff+1)])
        h=h_elem

# use the loop to seup the matrix h
    for k in range ((diff+1),input_layer):
        O=tf.constant([[1.]]) # first element for bias
        for i in range (1,diff):
            elem=tf.constant([[1.]])
            O=tf.concat([O,elem],1)
        for j in range (diff,input_layer):
            if j==k:
                elem=tf.constant([[1.]])
            else:
                elem=tf.constant([[0.]])
            O=tf.concat([O,elem],1)
        D=tf.diag(O[0])
        h_elem=tf.matmul(X,D)
        h_elem=tf.matmul(h_elem,W[:,(k-diff):(k-diff+1)])
        #print(h_elem.shape)
        h=tf.concat([h,h_elem],1)
	#print(h.shape)

    h0    = tf.nn.sigmoid(h)  # The \sigma function
    yhat = tf.matmul(h0, w_2)  # The \varphi function
    return yhat




def get_data():
    data=loadtxt("data.txt")
    target = loadtxt("target.txt")

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    all_Y = target

    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():

    f=open("result.txt","w+")


    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes
    h_size = 75                # Number of hidden nodes
    #h_size = 60                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights0((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # original 0.01

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
	avg_cost = 0.
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
	    avg_cost += sess.run(cost, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})/len(train_X)

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))


        f.write("%d, %.9f, %.2f%%, %.2f%%\n"  #epoch, cost, test accuracy, train accuracy
              % (epoch + 1, avg_cost, 100. * test_accuracy, 100. * train_accuracy))



    sess.close()

    f.write("test accuracy = %.2f%%, train accuracy = %.2f%%\n"
              % (100. * test_accuracy, 100. * train_accuracy))



    f.close()


if __name__ == '__main__':
    main()
