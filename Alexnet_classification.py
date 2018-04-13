import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import const_variables_list
import sys
sys.path.append(const_variables_list.path_to_input_pipeline_home)

import prepare_dataset as prepare
import readData as read

label_input_path = "/home/rashid/Projects/FlowerCounter/label/part-00000"
root_log_dir_for_tflog = "../../tf_logs"

# label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"

number_of_classes = 3
learning_rate = 0.00001
batch_size = 50

images_train,labels_train,images_test,labels_test = prepare.get_train_test_sets(label_input_path,train_ratio = 0.7,binning="True")

# A vector of filenames for trainset.
images_input_train = tf.constant(images_train)
images_labels_train = tf.constant(labels_train)

dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_labels_train))

Batched_dataset_train = dataset_train.shuffle(buffer_size=12000)\
                        .map(lambda x,y : read._parse_function(x,y,onehot = True, number_of_classes = number_of_classes))\
                        .batch(batch_size=batch_size)\
                        .repeat()

# Iterator for train dataset.
iterator_train = Batched_dataset_train.make_one_shot_iterator()

train_images,train_labels = iterator_train.get_next()

# A vector of filenames for testset.
images_input_test = tf.constant(images_test)
images_labels_test = tf.constant(labels_test)

dataset_test = tf.data.Dataset.from_tensor_slices((images_input_test, images_labels_test))
Batched_dataset_test = dataset_test.shuffle(buffer_size=4000)\
                        .map(lambda x,y : read._parse_function(x,y,onehot = True, number_of_classes = number_of_classes))\
                        .batch(batch_size=batch_size)\
                        .repeat()

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
last_layer = number_of_classes

# Fully connected densed layer - 1st
fc1 = tf.layers.dense(flattened, n_hidden1, name = "fc1", activation=tf.nn.relu, use_bias= True)

# Drop out for 1st fc layer
dropout1 = tf.nn.dropout(fc1, keep_prob)

# Fully connected densed layer - 2nd
fc2 = tf.layers.dense(dropout1, n_hidden2, name = "fc2",activation=tf.nn.relu, use_bias= True)

# Drop out for 2nd fc layer
dropout2 = tf.nn.dropout(fc2, keep_prob)

# Fully connected densed layer - 3rd and final
prediction = tf.layers.dense(dropout2,last_layer,name = "prediction",activation = None, use_bias= True)

# Place Holder for actual labels.
Y = tf.placeholder(tf.float32, shape= [None,number_of_classes],name = "labels")

# Op for calculating the loss
with tf.name_scope("cross_ent"):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# tf log initialization.
currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/run-{}/".format(root_log_dir_for_tflog,currenttime)

# # summary writter - Cost.
# cost_summary_train = tf.summary.scalar("Training loss", loss)
# cost_summary_test = tf.summary.scalar("Testing loss", loss)

# summary writter - Accuracy.
acc_summary_train = tf.summary.scalar("Training accuracy", accuracy)
acc_summary_test = tf.summary.scalar("Testing accuracy", accuracy)

file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())

num_steps = len(images_train)//batch_size
display_step = 20

validation_steps = len(images_test)//batch_size


init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    epoch = 0

    while (epoch < 50):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        for step in range(1, num_steps + 1):

            elem = sess.run([train_images, train_labels])

            # Allah = sess.run(Y,feed_dict={Y:elem[1]})
            # print(Allah)

            output = sess.run(train_op, feed_dict={X: elem[0],keep_prob: 0.4,Y: elem[1]})

            if (step % display_step == 0):

                # Collecting Trainset Accuracy.
                accur, accur_sum_train = sess.run([accuracy,acc_summary_train],feed_dict={X: elem[0],keep_prob: 1, Y: elem[1]} )

                print("training accuracy: {}".format(accur))

                # file_writer.add_summary(accur_sum_train, epoch*num_steps + step)


                # Collecting Testset Accuracy.
                test_elem = sess.run([test_images, test_labels])

                accur_test, accur_sum_test = sess.run([accuracy, acc_summary_test],
                                                  feed_dict={X: test_elem[0], keep_prob: 1, Y: test_elem[1]})

                print("validation accuracy: {}".format(accur_test))

                file_writer.add_summary(accur_sum_test, epoch * num_steps + step)

        epoch+=1

file_writer.close()