#coding:utf-8

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    """
     以正态分布生成随机值
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    初始化随机偏置
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    stride=1, padding=0, 
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling(s):
    """
    pooling层采用2 * 2 的
    """
    return tf.nn.max_pool(s, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])

#一层 5 * 5 卷积核, 输入通道1， 输出通道3
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print("h_conv1 shape:", h_conv1.shape)
h_pool1 = max_pooling(h_conv1)
print("h_pool1 shape:", h_pool1.shape)

#二层 5 * 5 卷积核，输入通道32，输出通道64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print("h_conv2 shape:", h_conv2.shape)
h_pool2 = max_pooling(h_conv2)
print("h_pool2 shape:", h_pool2.shape)

#7 * 7， 第三层全连接，输入是64通道，输出为1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y_conv))
#Adam优化器做最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(200):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x : batch[0], y_ : batch[1], keep_prob : 1.0
            })
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x : batch[0], y_ : batch[1], keep_prob : 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x : mnist.test.images, y_ : mnist.test.labels, keep_prob : 1.0
    }))

