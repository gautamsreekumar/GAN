import tensorflow as tf
import numpy as np
import numpy.random as nprand
import seaborn
import matplotlib.pyplot as plt
import cv2 as cv

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_dataset/", one_hot=True)

STD_DEV = 0.01

alpha = 0.2
alpha_g = alpha
g_input_layer = 100
g_output_layer = 784
g_hidden_layer = 128

z = tf.placeholder(tf.float32, shape=(None, g_input_layer))
# all variables are initialized with tf.random_normal()
g_W1 = tf.Variable(tf.random_normal([g_input_layer, g_hidden_layer], stddev=STD_DEV), name='g_W1')
g_b1 = tf.Variable(tf.zeros([g_hidden_layer]), name='g_b1')
g_W2 = tf.Variable(tf.random_normal([g_hidden_layer, g_output_layer], stddev=STD_DEV), name='g_W2')
g_b2 = tf.Variable(tf.zeros([g_output_layer]), name='g_b2')

alpha_d = 0.12
d_input_layer = 784
d_output_layer = 1
d_hidden_layer = 128

x = tf.placeholder(tf.float32, shape=(None, d_input_layer)) # None in the dimension is a place to fit in mini-batch
# all variables are initialized with tf.random_normal()
d_W1 = tf.Variable(tf.random_normal(shape=[d_input_layer, d_hidden_layer], stddev=STD_DEV), name='d_W1')
d_b1 = tf.Variable(tf.zeros(shape=[d_hidden_layer]), name='d_b1')
d_W2 = tf.Variable(tf.random_normal(shape=[d_hidden_layer, d_output_layer], stddev=STD_DEV), name='d_W2')
d_b2 = tf.Variable(tf.zeros(shape=[d_output_layer]), name='d_b2')

def g_net(z):
    g_layer1 = tf.nn.relu(tf.matmul(z, g_W1) + g_b1)
    g_y = tf.sigmoid(tf.matmul(g_layer1, g_W2) + g_b2)
    # return tf.reshape(g_y, [g_output_layer, 1])
    return g_y, g_layer1

def d_net(x):
    d_layer1 = tf.nn.relu(tf.matmul(x, d_W1) + d_b1)
    d_y = tf.sigmoid(tf.matmul(d_layer1, d_W2) + d_b2)
    return d_y, d_layer1

d_y, d_layer1 = d_net(x)
g_y, g_layer1 = g_net(z)
d_g_y, d_g_layer1 = d_net(g_y)
maximize_this = -tf.reduce_mean(tf.log(d_y)+tf.log(1-d_g_y), axis=0)
optim_d = [d_W1, d_b1, d_W2, d_b2] # list of variables to be updated in d-network
train_step_d = tf.train.GradientDescentOptimizer(alpha_d).minimize(maximize_this, var_list=optim_d)

minimize_this = -tf.reduce_mean(tf.log(d_g_y), axis=0)
optim_g = [g_W1, g_b1, g_W2, g_b2] # list of variables to updated in g-network
train_step_g = tf.train.GradientDescentOptimizer(alpha_g).minimize(minimize_this, var_list=optim_g)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.summary.scalar("Minimize_this", minimize_this[0])
tf.summary.scalar("Maximize_this", maximize_this[0])
merged_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./tensorflow_log')

# displaying the shapes of the matrices and vectors in G-network and D-network
print sess.run(tf.shape(d_W1)), sess.run(tf.shape(d_b1)), sess.run(tf.shape(d_W2)), sess.run(tf.shape(d_b2))
print sess.run(tf.shape(g_W1)), sess.run(tf.shape(g_b1)), sess.run(tf.shape(g_W2)), sess.run(tf.shape(g_b2))

number_of_iterations = 10000
k_steps = 2 # learning ratio of discriminator to that of the generator
batch_size = 50
sample_length = 10
test_cases = 1000
minibatch = 1

_ = 0
# for iteration in range(number_of_iterations):
while True:
    for k in range(k_steps):
        next_batch = mnist.train.next_batch(batch_size)
        sample_z = nprand.uniform(0, 1, size= [batch_size, g_input_layer])
        maximize_value = sess.run(maximize_this, feed_dict={x: next_batch[0], z: sample_z})
        minimize_value = sess.run(minimize_this, feed_dict={x: next_batch[0], z: sample_z})
        d_pred = sess.run(tf.reduce_mean(d_y, axis=0), feed_dict={x: next_batch[0], z: sample_z})
        g_pred = sess.run(tf.reduce_mean(d_g_y, axis=0), feed_dict={x: next_batch[0], z: sample_z})
        print maximize_value, minimize_value, d_pred, g_pred, _
        sess.run(train_step_d, feed_dict={x: next_batch[0], z:sample_z})
    sample_z = nprand.uniform(size=[batch_size, g_input_layer])
    minimize_value = sess.run(minimize_this, feed_dict={x: next_batch[0], z: sample_z})
    g_pred = sess.run(tf.reduce_mean(d_g_y, axis=0), feed_dict={x: next_batch[0], z: sample_z})
    print minimize_value, g_pred, _
    sess.run(train_step_g, feed_dict={z: sample_z})
    summary_str = sess.run(merged_ops, feed_dict={x: next_batch[0], z: sample_z})

    summary_writer.add_summary(summary_str, _)
    '''if np.abs(d_pred/(g_pred+d_pred) - 0.5) < epsilon: # breaking condition
        break
    if np.abs(maximize_value) == np.inf or np.abs(minimize_value) == np.inf or np.isnan(np.abs(maximize_value)) or np.isnan(np.abs(minimize_value)):
        break'''
    if _ % 100 == 0:
        duplicate = []
        sample_z = nprand.uniform(size=[batch_size, g_input_layer])
        duplicate = list(sess.run(g_y, feed_dict={z:sample_z}))
        digit = 255*np.asarray(duplicate[0]).reshape(28, 28)
        cv.imwrite('./digit_output/digit_'+str(_)+'.png', digit)
    _ = _ + 1

# testing

duplicate = []
sample_z = nprand.uniform(size=[batch_size, g_input_layer])
duplicate = list(sess.run(g_y, feed_dict={z:sample_z}))
digit = 255*np.asarray(duplicate[0]).reshape(28, 28)
cv.imwrite('digit.png', digit)
