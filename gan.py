import tensorflow as tf
import numpy as np
import numpy.random as nprand
import seaborn
import matplotlib.pyplot as plt

u_true = 10
sigma_true = 10
u_g = 1
sigma_g = 1

number_of_iterations = 1000
k_steps = 10 # learning ratio of discriminator to that of the generator
m = 100 # batch size
test_cases = 1000
minibatch = 100

alpha = 0.01
alpha_g = alpha
g_input_layer = m
g_output_layer = 1
g_hidden_layer = 2*m
g_outer = tf.nn.sigmoid
with tf.variable_scope("g_network", reuse=tf.AUTO_REUSE):
    z = tf.placeholder(tf.float64, shape=(g_input_layer, None))
    g_W1 = tf.get_variable('g_W1', [g_hidden_layer, g_input_layer], dtype=tf.float64)
    g_b1 = tf.get_variable('g_b1', [g_hidden_layer, minibatch], dtype=tf.float64)
    g_W2 = tf.get_variable('g_W2', [g_output_layer, g_hidden_layer], dtype=tf.float64)
    g_b2 = tf.get_variable('g_b2', [g_output_layer, minibatch], dtype=tf.float64)
    g_layer1 = tf.sigmoid(tf.matmul(g_W1, z)+g_b1)
    g_y = g_outer(tf.matmul(g_W2, g_layer1)+g_b2)

alpha_d = alpha
d_input_layer = 1
d_output_layer = 1
d_hidden_layer = m
d_outer = tf.nn.sigmoid

with tf.variable_scope("d_network", reuse=tf.AUTO_REUSE):
    x = tf.placeholder(tf.float64, shape=(d_input_layer, None)) # None in the dimension is a place to fit in
    # mini-batch
    d_W1 = tf.get_variable('d_W1', [d_hidden_layer, d_input_layer], dtype=tf.float64)
    d_b1 = tf.get_variable('d_b1', [d_hidden_layer, minibatch], dtype=tf.float64)
    d_W2 = tf.get_variable('d_W2', [d_output_layer, d_hidden_layer], dtype=tf.float64)
    d_b2 = tf.get_variable('d_b2', [d_output_layer, minibatch], dtype=tf.float64)
    d_layer1 = tf.sigmoid(tf.matmul(d_W1, x)+d_b1)
    d_y = d_outer(tf.matmul(d_W2, d_layer1)+d_b2)
    d_g_layer1 = tf.sigmoid(tf.matmul(d_W1, g_y)+d_b1)
    d_g_y = d_outer(tf.matmul(d_W2, d_g_layer1)+d_b2)

maximize_this = tf.reduce_mean(tf.reduce_sum(tf.log(d_y)+tf.log(1-d_g_y), axis=1))
optim_d = [d_W1, d_b1, d_W2, d_b2] # list of variables to be updated in d-network
grad_d = tf.gradients(xs=optim_d, ys=maximize_this)
d_W1_grad = optim_d[0].assign(optim_d[0] + alpha_d*grad_d[0]) # gradient ascent
d_b1_grad = optim_d[1].assign(optim_d[1] + alpha_d*grad_d[1]) # gradient ascent
d_W2_grad = optim_d[2].assign(optim_d[2] + alpha_d*grad_d[2]) # gradient ascent
d_b2_grad = optim_d[3].assign(optim_d[3] + alpha_d*grad_d[3]) # gradient ascent
update_d = [d_W1_grad, d_b1_grad, d_W2_grad, d_b2_grad]
train_step_d = tf.train.GradientDescentOptimizer(-alpha_d).minimize(maximize_this, var_list=optim_d)

minimize_this = tf.reduce_mean(tf.reduce_sum(tf.log(1-d_g_y), axis=1))
optim_g = [g_W1, g_b1, g_W2, g_b2] # list of variables to updated in g-network
grad_g = tf.gradients(ys=minimize_this, xs=optim_g)
g_W1_grad = optim_g[0].assign(optim_g[0] - alpha_g*grad_g[0]) # gradient descent
g_b1_grad = optim_g[1].assign(optim_g[1] - alpha_g*grad_g[1]) # gradient descent
g_W2_grad = optim_g[2].assign(optim_g[2] - alpha_g*grad_g[2]) # gradient descent
g_b2_grad = optim_g[3].assign(optim_g[3] - alpha_g*grad_g[3]) # gradient descent
update_g = [g_W1_grad, g_b1_grad, g_W2_grad, g_b2_grad]
train_step_g = tf.train.GradientDescentOptimizer(alpha_g).minimize(minimize_this, var_list=optim_g)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.summary.scalar("Minimize_this", minimize_this)
tf.summary.scalar("Maximize_this", maximize_this)
merged_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./tensorflow_log')

_ = 0
for iteration in range(number_of_iterations):
# while True:
    for k in range(k_steps):
        with tf.variable_scope("d_network", reuse=tf.AUTO_REUSE):
            sampled_x = nprand.normal(u_true, sigma_true, 1).reshape(1, 1)
            sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
            sess.run(update_d, feed_dict={x: sampled_x, z:sampled_z})
    with tf.variable_scope("g_network", reuse=tf.AUTO_REUSE):
        sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
        sess.run(update_g, feed_dict={z:sampled_z})
    summary_str = sess.run(merged_ops, feed_dict={x: sampled_x, z: sampled_z})
    summary_writer.add_summary(summary_str, _)
    maximize_value = sess.run(maximize_this, feed_dict={x: sampled_x, z: sampled_z})
    minimize_value = sess.run(minimize_this, feed_dict={x: sampled_x, z: sampled_z})
    print maximize_value, minimize_value, _
    if np.abs(maximize_value) == np.inf or np.abs(minimize_value) == np.inf or np.isnan(np.abs(maximize_value)) or np.isnan(np.abs(minimize_value)):
        break
    _ = _ + 1

# testing

duplicate = []
sampled_z = nprand.normal(u_g, sigma_g, (m, minibatch))
duplicate = list(sess.run(g_y, feed_dict={z:sampled_z}))
fig = seaborn.distplot(duplicate, hist=False) # gives error when minimize_value or maximum_value is NaN
plt.show()
