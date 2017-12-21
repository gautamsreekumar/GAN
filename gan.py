import tensorflow as tf
import numpy as np
import numpy.random as nprand

u_true = 10
sigma_true = 10
u_g = 1
sigma_g = 1

number_of_iterations = 1000
k_steps = 10 # learning ratio of discriminator to that of the generator
m = 100 # batch size
test_cases = 1000

alpha_g = 0.2
g_input_layer = m
g_output_layer = m
g_hidden_layer = 2*m
z = tf.placeholder(tf.float64, shape=(g_input_layer, None))
g_W1 = tf.Variable(tf.zeros([g_hidden_layer, g_input_layer], dtype=tf.float64), name='g_W1')
g_b1 = tf.Variable(tf.zeros([g_hidden_layer, 1], dtype=tf.float64), name='g_b1')
g_W2 = tf.Variable(tf.zeros([g_output_layer, g_hidden_layer], dtype=tf.float64), name='g_W2')
g_b2 = tf.Variable(tf.zeros([g_output_layer, 1], dtype=tf.float64), name='g_b2')
g_y = tf.nn.softmax(tf.matmul(g_W2, tf.matmul(g_W1, z)+g_b1)+g_b2)
tf.cast(g_y, tf.float64)

alpha_d = 0.2
d_input_layer = m
d_output_layer = 1
d_hidden_layer = 2*m
x = tf.placeholder(tf.float64, shape=(d_input_layer, None))
d_W1 = tf.Variable(tf.zeros([d_hidden_layer, d_input_layer], dtype=tf.float64), name='d_W1')
d_b1 = tf.Variable(tf.zeros([d_hidden_layer, 1], dtype=tf.float64), name='d_b1')
d_W2 = tf.Variable(tf.zeros([d_output_layer, d_hidden_layer], dtype=tf.float64), name='d_W2')
d_b2 = tf.Variable(tf.zeros([d_output_layer, 1], dtype=tf.float64), name='d_b2')
d_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, x)+d_b1)+d_b2)
d_g_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, g_y)+d_b1)+d_b2)

unity = tf.Variable(1.0, dtype=tf.float64)
maximize_this = tf.reduce_mean(tf.reduce_sum(tf.log(d_y)+tf.log(unity-d_g_y))) # the negative sign for
# gradient ascent has already been included
optim_d = [d_W1, d_b1, d_W2, d_b2]
grad_d = tf.train.GradientDescentOptimizer(alpha_d).compute_gradients(maximize_this, var_list=optim_d)
for item in range(len(optim_d)):
    optim_d[item].assign(optim_d[item] + alpha_d*grad_d[item][0]) # gradient ascent
# train_step_d = tf.train.GradientDescentOptimizer(alpha_d).minimize(maximize_this)

minimize_this = tf.reduce_mean(tf.reduce_sum(tf.log(unity-d_g_y)))
optim_g = [g_W1, g_b1, g_W2, g_b2]
grad_g = tf.train.GradientDescentOptimizer(alpha_g).compute_gradients(minimize_this, var_list=optim_g)
for item in range(len(optim_g)):
    optim_g[item].assign(optim_g[item] - alpha_g*grad_g[item][0]) # gradient descent
# train_step_g = tf.train.GradientDescentOptimizer(alpha_g).minimize(minimize_this)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run(session=sess)

print [v.name for v in tf.trainable_variables()]
for iterations in range(number_of_iterations):
    for k in range(k_steps):
        sampled_x = nprand.normal(u_true, sigma_true, m).reshape(m, 1)
        sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
        # print sess.run(train_step_d, feed_dict={x: sampled_x, z:sampled_z}), "training discriminator"
        sess.run(grad_d, feed_dict={x: sampled_x, z:sampled_z})
        print "training discriminator"
        
    sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
    # print sess.run(train_step_g, feed_dict={z: sampled_z}), "training generator"
    sess.run(grad_g, feed_dict={z:sampled_z})
    print "training generator"

print sess.run(d_W2)

for testing in range(test_cases):
    sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
    print "testing", sess.run(d_g_y, feed_dict={z:sampled_z})
