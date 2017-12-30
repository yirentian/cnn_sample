#coding:utf-8

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./", "Directory for storing data")

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    """
    随机生成weight
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    随机生成bias
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    卷积层，stride=1, padding=0
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling(s):
    """
    池华层，stride=2
    """
    return tf.nn.max_pool(s, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])

#layer 1 : conv 5 * 5, input channel 1, output channel 6
W_conv1 = weight_variable([5, 5, 1, 6])
bias_conv1 = bias_variable([6])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + bias_conv1)
h_pool1 = max_pooling(h_conv1)
print("hidden layer1 shape={}:".format(h_pool1.shape))

#layer 2 : conv 5 : 5 , input channel 6, output channel 6
W_conv2 = weight_variable([5, 5, 6, 16])
bias_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + bias_conv2)
h_pool2 = max_pooling(h_conv2)
print("hidden layer2 shape={}".format(h_pool2.shape))

#full connected layer3
W_fc1 = weight_variable([7 * 7 * 16, 120])
bias_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + bias_fc1)
print("hidden layer3 shape={}".format(h_fc1.shape))

#full connected layer4
W_fc2 = weight_variable([120, 84])
bias_fc2 = bias_variable([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + bias_fc2)
keep_prob = tf.placeholder(tf.float32)
h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob)
print("hidden layer4 shape={}".format(h_fc2.shape))

#softmax
W_fc3 = weight_variable([84, 10])
bias_fc3 = weight_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc2_dropout, W_fc3) + bias_fc3)
print("output layer shape={}".format(y.shape))

#loss function
cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

#optimize
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 200 == 0:
        print("step {} accuracy: {}".format(i, accuracy.eval(feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0})))
    sess.run(train_step, feed_dict={x : batch[0], y_ : batch[1], keep_prob : 0.5})

#test
print("test accuracy :{}".format(accuracy.eval(feed_dict={x : mnist.test.images, y_ : mnist.test.labels, keep_prob : 1.0})))

#save
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
saver.save(sess, "./models.ckpt")

