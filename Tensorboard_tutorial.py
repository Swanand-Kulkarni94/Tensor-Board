#Program 9, Use of Tensorboard while using a Tensorflow

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

tf.set_random_seed(1)
np.random.seed(1)

#Generate dumy data
x = np.linspace(-1, 1, 100)[:, np.newaxis] #Shape (100, 1)
noise = np.random.normal(0, 0.1, size = x.shape)
y = np.power(x, 2) + noise

with tf.variable_scope('Inputs'):
	tf_x = tf.placeholder(tf.float32, x.shape, name = 'x')
	tf_y = tf.placeholder(tf.float32, y.shape, name = 'y')

with tf.variable_scope('Net'):
	layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name = 'Hidden_Layer')
	output = tf.layers.dense(layer1, 1, name = 'Output_Layer')

	#Add to histogram summary
	tf.summary.histogram('h_out', layer1)
	tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(tf_y, output, scope = 'loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(loss)
tf.summary.scalar('loss', loss) #Add loss to scalar summary

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('C:\#DATA\Work\Sublime_Programs\Tensorboard/log', sess.graph)
merge_op = tf.summary.merge_all()

for step in range(100):
	#Train and obtain output
	_, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
	writer.add_summary(result, step)

