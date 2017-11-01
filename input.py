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


#This function reads single image file and converts it into a

def test_read_image(image):
    sess = tf.InteractiveSession()
    fn = image
    image_contents = tf.read_file(fn)
    im = tf.image.decode_image(image_contents, channels=3)
    im = tf.image.resize_image_with_crop_or_pad(im, 500, 500)
    print(im.shape)
    plt.imshow(im.eval())
    plt.show()
    sess.close()

test_read_image('flower.jpg')
