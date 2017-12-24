import tensorflow as tf
import numpy as np
import numpy.random as nprand

u_true = 10
sigma_true = 10
u_g = 1
sigma_g = 1

number_of_iterations = 100
k_steps = 10 # learning ratio of discriminator to that of the generator
m = 100 # batch size
test_cases = 1000

alpha = 0.01
alpha_g = alpha
g_input_layer = m
g_output_layer = m
g_hidden_layer = 2*m
with tf.variable_scope("g_network", reuse=tf.AUTO_REUSE):
    z = tf.placeholder(tf.float64, shape=(g_input_layer, None))
    '''g_W1 = tf.get_variable(tf.zeros([g_hidden_layer, g_input_layer], dtype=tf.float64), name='g_W1')
    g_b1 = tf.get_variable(tf.zeros([g_hidden_layer, 1], dtype=tf.float64), name='g_b1')
    g_W2 = tf.get_variable(tf.zeros([g_output_layer, g_hidden_layer], dtype=tf.float64), name='g_W2')
    g_b2 = tf.get_variable(tf.zeros([g_output_layer, 1], dtype=tf.float64), name='g_b2')'''
    g_W1 = tf.get_variable('g_W1', [g_hidden_layer, g_input_layer], dtype=tf.float64)
    g_b1 = tf.get_variable('g_b1', [g_hidden_layer, 1], dtype=tf.float64)
    g_W2 = tf.get_variable('g_W2', [g_output_layer, g_hidden_layer], dtype=tf.float64)
    g_b2 = tf.get_variable('g_b2', [g_output_layer, 1], dtype=tf.float64)
    g_y = tf.nn.softmax(tf.matmul(g_W2, tf.matmul(g_W1, z)+g_b1)+g_b2)

alpha_d = alpha
d_input_layer = m
d_output_layer = 1
d_hidden_layer = 2*m
with tf.variable_scope("d_network", reuse=tf.AUTO_REUSE):
    x = tf.placeholder(tf.float64, shape=(d_input_layer, None))
    '''d_W1 = tf.get_variable(tf.zeros([d_hidden_layer, d_input_layer], dtype=tf.float64), name='d_W1')
    d_b1 = tf.get_variable(tf.zeros([d_hidden_layer, 1], dtype=tf.float64), name='d_b1')
    d_W2 = tf.get_variable(tf.zeros([d_output_layer, d_hidden_layer], dtype=tf.float64), name='d_W2')
    d_b2 = tf.get_variable(tf.zeros([d_output_layer, 1], dtype=tf.float64), name='d_b2')
    d_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, x)+d_b1)+d_b2)
    d_g_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, g_y)+d_b1)+d_b2)'''
    d_W1 = tf.get_variable('d_W1', [d_hidden_layer, d_input_layer], dtype=tf.float64)
    d_b1 = tf.get_variable('d_b1', [d_hidden_layer, 1], dtype=tf.float64)
    d_W2 = tf.get_variable('d_W2', [d_output_layer, d_hidden_layer], dtype=tf.float64)
    d_b2 = tf.get_variable('d_b2', [d_output_layer, 1], dtype=tf.float64)
    d_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, x)+d_b1)+d_b2)
    d_g_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, g_y)+d_b1)+d_b2)

unity = tf.Variable(1.0, dtype=tf.float64)
maximize_this = tf.reduce_mean(tf.reduce_sum(tf.log(d_y)+tf.log(unity-d_g_y)))
optim_d = [d_W1, d_b1, d_W2, d_b2] # list of variables to be updated in d-network
grad_d = tf.train.GradientDescentOptimizer(alpha_d).compute_gradients(maximize_this, var_list=optim_d)
for item in range(len(optim_d)):
    optim_d[item].assign(optim_d[item] + alpha_d*grad_d[item][0]) # gradient ascent
'''d_W1.assign(d_W1 + alpha_d*grad_d[0][0])
d_b1.assign(d_b1 + alpha_d*grad_d[1][0])
d_W2.assign(d_W2 + alpha_d*grad_d[2][0])
d_b2.assign(d_b2 + alpha_d*grad_d[3][0])'''
train_step_d = tf.train.GradientDescentOptimizer(alpha_d).minimize(-maximize_this, var_list=optim_d)

minimize_this = tf.reduce_mean(tf.reduce_sum(tf.log(unity-d_g_y)))
optim_g = [g_W1, g_b1, g_W2, g_b2] # list of variables to updated in g-network
grad_g = tf.train.GradientDescentOptimizer(alpha_g).compute_gradients(minimize_this, var_list=optim_g)
for item in range(len(optim_g)):
    optim_g[item].assign(optim_g[item] - alpha_g*grad_g[item][0])
'''g_W1.assign(g_W1 - alpha_g*grad_g[0][0])
g_b1.assign(g_b1 - alpha_g*grad_g[1][0])
g_W2.assign(g_W2 - alpha_g*grad_g[2][0])
g_b2.assign(g_b2 - alpha_g*grad_g[3][0])'''
train_step_g = tf.train.GradientDescentOptimizer(alpha_g).minimize(minimize_this, var_list=optim_g)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for iterations in range(number_of_iterations):
    for k in range(k_steps):
        with tf.variable_scope("d_network", reuse=tf.AUTO_REUSE):
            sampled_x = nprand.normal(u_true, sigma_true, m).reshape(m, 1)
            sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
            print sess.run(train_step_d, feed_dict={x: sampled_x, z:sampled_z}), "training discriminator"
            # sess.run(optim_d, feed_dict={x: sampled_x, z:sampled_z})
            # print "training discriminator"
    with tf.variable_scope("g_network", reuse=tf.AUTO_REUSE):
        sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
        print sess.run(train_step_g, feed_dict={z: sampled_z}), "training generator"
        # sess.run(optim_g, feed_dict={z:sampled_z})
        # print "training generator"

for testing in range(test_cases):
    sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
    print "testing", sess.run(d_g_y, feed_dict={z:sampled_z})

print sess.run(g_W2)
