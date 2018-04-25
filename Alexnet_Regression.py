import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import const_variables_list
import sys
sys.path.append(const_variables_list.path_to_input_pipeline_discus)

import prepare_dataset as prepare
import readData as read

label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels"
root_log_dir_for_tflog = "../../tf_logs"

learning_rate = 0.00001
batch_size = 50

images_train,labels_train,images_test,labels_test = prepare.get_train_test_sets(label_input_path,train_ratio = 0.7)

train_img_names_op = tf.placeholder(tf.string, shape=[None])
train_label_op = tf.placeholder(tf.string, shape=[None])

dataset_train = tf.data.Dataset.from_tensor_slices((train_img_names_op, train_label_op))

Batched_dataset_train = dataset_train.shuffle(buffer_size=12000)\
                        .map(read._parse_function)\
                        .batch(batch_size=batch_size)\
                        .repeat()

# Iterator for train dataset.
iterator_train = Batched_dataset_train.make_initializable_iterator()

train_images,train_labels = iterator_train.get_next()

init_op = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init_op)

    sess.run(iterator_train.initializer, {train_img_names_op:images_train,
                                                    train_label_op:labels_train})

