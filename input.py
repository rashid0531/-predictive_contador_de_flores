'''
Author: Mohammed Rashid Chowdhury
Date: October 31st, 2017

This script writes the images into a TFRecord file.

This script is influenced by the following blogpost:

http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
'''

import numpy as np
import tensorflow as tf

#This function reads single image file and converts it into a

def read_image(image):
    img = tf.image.decode_jpeg(image,channels=3)
    img_resized = tf.image.resize_images(image,[299,299])
