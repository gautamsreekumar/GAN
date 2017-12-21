import tensorflow as tf
import numpy as np
import numpy.random as nprand

'''
class G():
    def __init__(self, m, alpha_g):
        self.alpha = alpha_g
        self.input_layer_size = m
        self.output_layer_size = 1
        self.hidden_layer_size = 2*m
        self.z = tf.placeholder(tf.float32, shape=(self.input_layer_size, None))
        self.W1 = tf.Variable(tf.zeros([self.hidden_layer_size, self.input_layer_size]))
        self.b1 = tf.Variable(tf.zeros([self.hidden_layer_size, 1]))
        self.W2 = tf.Variable(tf.zeros([self.output_layer_size, self.hidden_layer_size]))
        self.b2 = tf.Variable(tf.zeros([self.output_layer_size, 1]))
        self.y = tf.nn.softmax(tf.matmul(self.W2, tf.matmul(self.W1, self.z)+self.b1)+self.b2)

    def run_nn(self, input_z, update_variables, sess):
        tf.global_variables_initializer().run(session=sess)
        if update_variables == 1:
            minimize_this = tf.reduce_mean(tf.reduce_sum(tf.log(1-self.y)))
            train_step = tf.train.GradientDescentOptimizer(self.alpha).minimize(minimize_this)
            train_step.run(session=sess, feed_dict={self.z: input_z})
        else:
            return self.y

class D():
    def __init__(self, m, alpha_d):
        self.input_layer_size = m
        self.output_layer_size = 1
        self.hidden_layer_size = 2*m
        self.alpha = alpha_d
        self.x = tf.placeholder(tf.float32, shape=(self.input_layer_size, None))
        self.W1 = tf.Variable(tf.zeros([self.hidden_layer_size, self.input_layer_size]))
        self.b1 = tf.Variable(tf.zeros([self.hidden_layer_size, 1]))
        self.W2 = tf.Variable(tf.zeros([self.output_layer_size, self.hidden_layer_size]))
        self.b2 = tf.Variable(tf.zeros([self.output_layer_size, 1]))
        self.y = tf.nn.softmax(tf.matmul(self.W2, tf.matmul(self.W1, self.x)+self.b1)+self.b2)

    def run_nn(self, input_x, y_z, sess):
        tf.global_variables_initializer().run(session=sess)
        unity = tf.Variable(1.0, tf.float32)
        maximize_this = -1*tf.reduce_mean(tf.reduce_sum(tf.log(self.y)+tf.log(unity-y_z))) # the negative sign for
        # gradient ascent has already been included
        train_step = tf.train.GradientDescentOptimizer(self.alpha).minimize(maximize_this)
        train_step.run(session=sess, feed_dict={self.x: input_x})

u_true = 10
sigma_true = 10
u_g = 1
sigma_g = 1

number_of_iterations = 1000
k_steps = 10 # learning ratio of discriminator to that of the generator
m = 100 # batch size
alpha = 0.2

gen = G(m, alpha)
dis = D(m, alpha)
sess = tf.Session()

for iterations in range(number_of_iterations):
    for k in range(k_steps):
        x = list(nprand.normal(u_true, sigma_true, m).astype(np.float32).reshape(m, 1))
        z = list(nprand.normal(u_g, sigma_g, m).astype(np.float32).reshape(m, 1))
        y = gen.run_nn(z, 0, sess)
        dis.run_nn(x, y, sess)
'''

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
g_W1 = tf.Variable(tf.zeros([g_hidden_layer, g_input_layer], dtype=tf.float64))
g_b1 = tf.Variable(tf.zeros([g_hidden_layer, 1], dtype=tf.float64))
g_W2 = tf.Variable(tf.zeros([g_output_layer, g_hidden_layer], dtype=tf.float64))
g_b2 = tf.Variable(tf.zeros([g_output_layer, 1], dtype=tf.float64))
g_y = tf.nn.softmax(tf.matmul(g_W2, tf.matmul(g_W1, z)+g_b1)+g_b2)
tf.cast(g_y, tf.float64)

alpha_d = 0.2
d_input_layer = m
d_output_layer = 1
d_hidden_layer = 2*m
x = tf.placeholder(tf.float64, shape=(d_input_layer, None))
d_W1 = tf.Variable(tf.zeros([d_hidden_layer, d_input_layer], dtype=tf.float64))
d_b1 = tf.Variable(tf.zeros([d_hidden_layer, 1], dtype=tf.float64))
d_W2 = tf.Variable(tf.zeros([d_output_layer, d_hidden_layer], dtype=tf.float64))
d_b2 = tf.Variable(tf.zeros([d_output_layer, 1], dtype=tf.float64))
d_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, x)+d_b1)+d_b2)
d_g_y = tf.nn.softmax(tf.matmul(d_W2, tf.matmul(d_W1, g_y)+d_b1)+d_b2)

unity = tf.Variable(1.0, dtype=tf.float64)
maximize_this = -1*tf.reduce_mean(tf.reduce_sum(tf.log(d_y)+tf.log(unity-d_g_y))) # the negative sign for
# gradient ascent has already been included
train_step_d = tf.train.GradientDescentOptimizer(alpha_d).minimize(maximize_this)

minimize_this = tf.reduce_mean(tf.reduce_sum(tf.log(unity-d_g_y)))
train_step_g = tf.train.GradientDescentOptimizer(alpha_g).minimize(minimize_this)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run(session=sess)

for iterations in range(number_of_iterations):
    for k in range(k_steps):
        '''sampled_x = list(nprand.normal(u_true, sigma_true, m).reshape(m, 1))
        sampled_z = list(nprand.normal(u_g, sigma_g, m).reshape(m, 1))'''
        sampled_x = nprand.normal(u_true, sigma_true, m).reshape(m, 1)
        sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
        print sess.run(train_step_d, feed_dict={x: sampled_x, z:sampled_z}), "training discriminator"
    # sampled_z = list(nprand.normal(u_g, sigma_g, m).reshape(m, 1))
    sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
    print sess.run(train_step_g, feed_dict={z: sampled_z}), "training generator"

for testing in range(test_cases):
    sampled_z = nprand.normal(u_g, sigma_g, m).reshape(m, 1)
    print "testing", sess.run(d_g_y, feed_dict={z:sampled_z})
