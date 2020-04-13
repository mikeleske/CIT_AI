import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# create dataset
W, b = 0.5, 1.4
X = np.linspace(0,100, num=100)
y = np.random.normal(loc=W * X + b, scale=2.0, size=len(X))

# create the placeholders
x_ph = tf.placeholder(shape=[None,], dtype=tf.float32)
y_ph = tf.placeholder(shape=[None,], dtype=tf.float32)
# create the variables
v_weight = tf.get_variable("weight", shape=[1], dtype=tf.float32)
v_bias = tf.get_variable("bias", shape=[1], dtype=tf.float32)

# linear computation
out = v_weight * x_ph + v_bias
# compute the mean squared error
loss = tf.reduce_mean((out - y_ph)**2)

# optimise loss
opt = tf.train.AdamOptimizer(0.4).minimize(loss)

# create TF session and initialise variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# TensorBoard metrics
tf.summary.scalar('MSEloss', loss)
tf.summary.histogram('model_weight', v_weight)
tf.summary.histogram('model_bias', v_bias)
all_summary = tf.summary.merge_all()

# TensorBoard FileWriter
now = datetime.now()
clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
file_writer = tf.summary.FileWriter('log_dir/'+clock_time, session.graph)

# loop to train the parameters
for ep in range(210):
	# run the optimiser and get the loss
	train_loss, _, train_summary = session.run([loss, opt, all_summary], feed_dict={x_ph:X, y_ph:y})
	# write to TF event file (for TensorBoard)	
	file_writer.add_summary(train_summary, ep)
	# print epoch number and loss
	if ep % 40 == 0:
		print('Epoch: %3d, MSE: %.4f, W: %.3f, b: %.3f' % (ep, train_loss, session.run(v_weight), session.run(v_bias)))

# print final model parameters
print('Final weight: %.3f, bias: %.3f' % (session.run(v_weight), session.run(v_bias)))

# plot
plt.scatter(X,y)
plt.plot(X,session.run(out, feed_dict={x_ph:X}),color='r')
plt.show()

# close FileWriter
file_writer.close()

