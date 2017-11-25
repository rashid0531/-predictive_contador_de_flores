import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mil
from random import shuffle
import glob
import create_TFRecord
from PIL import Image
from datetime import datetime

import input
import create_TFRecord
import CNN_models.Alexnet.alexnet as cnnmodel


# Input file
path = "../test_set/original.txt"

# Parameters for initializing dataset.
train_percent = 0.7
test_percent = 0.3

# Fetching training & testing set.
train_set, train_label, test_set, test_label = input.get_train_test_validation_sets(path=path,train_percent=train_percent,test_percent=test_percent)
print(len(train_set)+len(test_set))

# Read the image data as numpy array.
train_set = list(map(input.read_image_using_PIL,train_set))

# Creating TFRecords for training set.
tfRecord_name = 'train.tfrecords'
create_TFRecord.stash_example_protobuff_to_tfrecord(tfRecord_name,train_set,train_label)

# Training params for learning the model
learning_rate = 0.005
num_epochs = 1
batch_size = 5

# Network parameters
dropout_rate = 0.5
num_classes = 5
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 10

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = cnnmodel.AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(len(train_set)/batch_size))
val_batches_per_epoch = int(np.floor(len(test_set)/ batch_size))

print(train_batches_per_epoch)

# Start Tensorflow session
with tf.Session() as sess:

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    image, label = create_TFRecord.read_singleExample_tfrecord(tfRecord_name)
    # To simply avoid queueing errors set allow_smaller_final_batch to 'True'
    image_batches, label_batches = tf.train.batch([image, label], batch_size=batch_size, num_threads=4,
                                                  allow_smaller_final_batch=True)

    # Initialize all variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        # sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            print("step :",step)
            # get next batch of data
            img_batch, label_batch = sess.run([image_batches,label_batches])

            # Converting the labels to one-hot encoding.

            label_batch = tf.one_hot(label_batch, 5)
            label_batch = sess.run(label_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,y: label_batch,keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)


    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    print("Done")


'''
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
    
'''