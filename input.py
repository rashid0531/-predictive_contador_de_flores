'''
Author: Mohammed Rashid Chowdhury
Date: October 31st, 2017

This script writes the images into a TFRecord file.

This script is influenced by the following blogpost:

http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mil
from random import shuffle
import glob
import create_TFRecord
from PIL import Image

#This function reads single image file and converts it into a

def test_read_image(image):

    sess = tf.InteractiveSession()
    fn = image
    image_contents = tf.read_file(fn)
    im = tf.image.decode_image(image_contents, channels=3)
    im = tf.image.resize_image_with_crop_or_pad(im, 500, 700)
    print(im.shape)
    plt.imshow(im.eval())
    plt.show()
    sess.close()

#test_read_image('flower_plots/flower.jpg')

def read_image(image):

    fn = image
    image_contents = tf.read_file(fn)
    im = tf.image.decode_image(image_contents, channels=3)
    im = tf.image.resize_image_with_crop_or_pad(im, 500, 700)
    # need to create a session and evaluate the decoded image by running it through session otherwise the data wont be converted to numpy array but as a tensor.
    sess = tf.InteractiveSession()
    im = sess.run(im)
    sess.close()

    return im


def read_image_using_PIL(image):

    image = Image.open(image)
    image = image.resize((250,250))
    image = np.asarray(image, np.uint8)
    print(image)
    print(image.shape)
    img = Image.fromarray(image,'RGB')
    img.show()
    # image = np.asarray(image, np.uint8)
    # shape = np.array(image.shape, np.int32)
    # return image.tobytes(),shape.tobytes()
    #return image.tobytes()

read_image_using_PIL('flower_plots/violet_blossom.jpg')


def get_bin(value):

    '''
    This function returns bin number for a given value. The process of defining the bin is subjected to be changed over time.
    For the time being, we agreed to start with 5 classes: Bin0, Bin1, Bin2, Bin3, Bin4.
    '''

    if (int(value) in range(0,700)):
        return "Bin0"

    elif (int(value) in range(700,800)):
        return "Bin1"

    elif (int(value) in range(800,900)):
        return "Bin2"

    elif (int(value) in range(900,1000)):
        return "Bin3"

    else:
        return "Bin4"


def list_imageData_with_labels(directory):

    '''
    This function creates label for each image files for the given directory.

    :param:
    -directory: The path where our data is saved.

    :return:
    -file_name: a list that contains the image names.
    -labels: a list of labels for correspondent image file

    '''
    labels =[]
    file_name = []
    path = "../test_set/flower_plots/"

    # for testing small dataset
    path = "flower_plots/"

    with open(directory) as file_object:

        for each_line in file_object:
            image_name = str(path+each_line.split( )[0])
            count = each_line.split( )[1]
            file_name.append(image_name)
            labels.append(get_bin(count))

    return file_name,labels

#files,labels = list_imageData_with_labels("../test_set/original.txt")

# For small dataset to test in a local machine
# files,labels = list_imageData_with_labels("original.txt")
# files = list(map(read_image,files))
# tfRecord_name = 'train.tfrecords'
# create_TFRecord.stash_example_protobuff_to_tfrecord(tfRecord_name,files,labels)
#
# # Ploting results received after transforming the tfrecord back to previous input format (numpy array)
# plt.imshow(create_TFRecord.read_tfrecords(tfRecord_name))
# plt.show()
