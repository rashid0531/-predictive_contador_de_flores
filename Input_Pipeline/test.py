import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

import Input_Pipeline.prepare_dataset as prepare
#import prepare_dataset as prepare

import Input_Pipeline.readData as read
#import readData as read


label_input_path = "/home/rashid/Projects/FlowerCounter/label/part-00000"
root_log_dir_for_tflog = "../../tf_logs"

# label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"

learning_rate = 0.0001
batch_size = 20

images_train,labels_train,images_test,labels_test = prepare.get_train_test_sets(label_input_path,train_ratio = 0.7)

# print(len(images_train), len(images_test))


# A vector of filenames for trainset.
images_input_train = tf.constant(images_train)
images_labels_train = tf.constant(labels_train)


dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_labels_train))

Batched_dataset_train = dataset_train.shuffle(buffer_size=12000).map(read._parse_function).batch(batch_size=batch_size).repeat()

# Iterator for train dataset.
iterator_train = Batched_dataset_train.make_one_shot_iterator()

train_images,train_labels = iterator_train.get_next()

# A vector of filenames for testset.
images_input_test = tf.constant(images_test)
images_labels_test = tf.constant(labels_test)

dataset_test = tf.data.Dataset.from_tensor_slices((images_input_test, images_labels_test))
Batched_dataset_test = dataset_test.shuffle(buffer_size=4000).map(read._parse_function).batch(batch_size=batch_size).repeat()

# Iterator for test dataset.
iterator_test = Batched_dataset_test.make_one_shot_iterator()

test_images,test_labels = iterator_test.get_next()

# Parameters of LRN.
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0

# Place holder for dropout percentage.
keep_prob = tf.placeholder(tf.float32, name= "dropput_rate")

# Information about the images.
height = 224
width = 224
channels = 3

# Placeholder for batched image data.
X = tf.placeholder(shape=(None,height,width,channels),dtype=tf.float32)

# First convulational layer
conv1 = tf.layers.conv2d(X, filters=96, kernel_size=11, strides=[4,4], padding="SAME",activation=tf.nn.relu)

# Local Response Normalization -1st
lrn1 = tf.nn.local_response_normalization(conv1,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

# Max pool layer - 1st
max_pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

# second convulational layer
conv2 = tf.layers.conv2d(max_pool1, filters=256, kernel_size=5, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Local Response Normalization -2nd
lrn2 = tf.nn.local_response_normalization(conv2,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

# Max pool layer - 2nd
max_pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

# Third convulational layer
conv3 = tf.layers.conv2d(max_pool2, filters=384, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Fourth convulational layer
conv4 = tf.layers.conv2d(conv3, filters=384, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Fifth convulational layer
conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3, strides=[1,1], padding="SAME",activation=tf.nn.relu)

# Max pool layer - 3rd
max_pool3 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")


# need to make those tensor a batch of 1D tensor.
flattened = tf.reshape(max_pool3, [-1, 6 * 6 * 256])

n_hidden1 = 4096
n_hidden2 = 4096
last_layer = 1

# Fully connected densed layer - 1st
fc1 = tf.layers.dense(flattened, n_hidden1, name = "fc1", activation=tf.nn.relu, use_bias= True)

# Drop out for 1st fc layer
dropout1 = tf.nn.dropout(fc1, keep_prob)

# Fully connected densed layer - 2nd
fc2 = tf.layers.dense(dropout1, n_hidden2, name = "fc2",activation=tf.nn.relu, use_bias= True)

# Drop out for 2nd fc layer
dropout2 = tf.nn.dropout(fc2, keep_prob)

# Fully connected densed layer - 3rd
fc3 = tf.layers.dense(dropout2, last_layer, name = "fc3",activation=None)

# Place Holder for actual labels.
Y = tf.placeholder(tf.float32,name = "labels")

with tf.name_scope("loss"):

    # Designing simple cost function.

    cost = tf.sqrt(tf.reduce_mean(tf.square(Y - fc3)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)


# tf log initialization.
currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/run-{}/".format(root_log_dir_for_tflog,currenttime)

# summary writter.
cost_summary_train = tf.summary.scalar("Training loss", cost)

cost_summary_test = tf.summary.scalar("Testing loss", cost)


file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())


# train_writer = tf.train.SummaryWriter(logdir + '/train', tf.get_default_graph())
# test_writer = tf.train.SummaryWriter(logdir + '/test',tf.get_default_graph())


num_steps = len(images_train)//batch_size
display_step = 50

validation_steps = len(images_test)//batch_size

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    epoch = 0

    while (epoch < 35):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        for step in range(1, num_steps + 1):

            elem = sess.run([train_images, train_labels])

            original_labels = np.reshape(elem[1], (-1, 1))

            output,loss = sess.run([train_op,cost],feed_dict={X:elem[0],
                                            keep_prob:0.5,
                                            Y:original_labels})

            if (step % display_step == 0):
                # Loss per epoch - testset
                test_elem = sess.run([test_images, test_labels])

                original_labels_test = np.reshape(test_elem[1], (-1, 1))

                test_cost_sum,test_loss = sess.run([cost_summary_test,cost], feed_dict={X: test_elem[0],
                                                                       keep_prob: 1,
                                                                       Y: original_labels_test})
                print("validation loss: {}".format(test_loss))

                file_writer.add_summary(test_cost_sum, epoch*num_steps + step)

        # # Loss per epoch - trainset
        #
        # train_elem = sess.run([train_images, train_labels])
        #
        # original_labels_train = np.reshape(train_elem[1], (-1, 1))
        #
        # train_cost_sum = sess.run(cost_summary_train, feed_dict={X: test_elem[0],
        #                                                   keep_prob: 1,
        #                                                   Y: original_labels_test})
        #
        # file_writer.add_summary(train_cost_sum)

        epoch+=1


file_writer.close()


