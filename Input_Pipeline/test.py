import tensorflow as tf
import glob
from matplotlib import pyplot as plt

import Input_Pipeline.readData as read
#import readData as read


label_input_path = "/home/rashid/Projects/FlowerCounter/label/part-00000"
images,labels = read.process_label_files(label_input_path)

# A vector of filenames.
images_input = tf.constant(images)
images_labels = tf.constant(labels)

dataset = tf.data.Dataset.from_tensor_slices((images_input, images_labels))
dataset = dataset.map(read._parse_function).batch(4)

iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()

images,labels = iterator.get_next()

# Information about the images.
height = 224
width = 224
channels = 3

# Placeholder for batched image data.
X = tf.placeholder(shape=(None,height,width,channels),dtype=tf.float32)

# First convulational layer
conv1 = tf.layers.conv2d(X, filters=96, kernel_size=11, strides=[4,4], padding="SAME",activation=tf.nn.relu)

# Max pool layer - 1st
max_pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

# second convulational layer
conv2 = tf.layers.conv2d(max_pool1, filters=256, kernel_size=5, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Max pool layer - 2nd
max_pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

# Third convulational layer
conv3 = tf.layers.conv2d(max_pool2, filters=384, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Fourth convulational layer
conv4 = tf.layers.conv2d(conv3, filters=384, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Fifth convulational layer
conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)


n_hidden1 = 4096
n_hidden2 = 4096

# Fully connected densed layer - 1st
fc1 = tf.layers.dense(conv5, n_hidden1, name = "fc1",activation=tf.nn.relu)

# Fully connected densed layer - 2nd
fc2 = tf.layers.dense(fc1, n_hidden2, name = "fc2",activation=tf.nn.relu)



init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    elem = sess.run([images, labels])

    output = sess.run(fc2,feed_dict={X:elem[0]})

print(output.shape)
plt.imshow(output[0,:,:,1])
plt.show()
