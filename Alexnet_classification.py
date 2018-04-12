import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import const_variables_list
import sys
sys.path.append(const_variables_list.path_to_input_pipeline)

import prepare_dataset as prepare
import readData as read


label_input_path = "/home/rashid/Projects/FlowerCounter/label/part-00000"
root_log_dir_for_tflog = "../../tf_logs"

# label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"

number_of_classes = 3
learning_rate = 0.0001
batch_size = 20

images_train,labels_train,images_test,labels_test = prepare.get_train_test_sets(label_input_path,train_ratio = 0.7,binning="True")

# A vector of filenames for trainset.
images_input_train = tf.constant(images_train)
images_labels_train = tf.constant(labels_train)

dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_labels_train))

Batched_dataset_train = dataset_train\
                        .map(lambda x,y : read._parse_function(x,y,onehot = True, number_of_classes = number_of_classes))\
                        .batch(batch_size=batch_size)\
                        .repeat()

# Iterator for train dataset.
iterator_train = Batched_dataset_train.make_one_shot_iterator()

train_images,train_labels = iterator_train.get_next()

init_op = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init_op)
    elem = sess.run(train_labels)
    print(elem)

