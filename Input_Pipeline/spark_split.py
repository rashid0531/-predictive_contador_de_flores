'''
Author: Mohammed Rashid Chowdhury
email:  <mrc689@mail.usask.ca>
Script name: create_tiles_skipping_crop.py
'''

# This script merges crop.py and slicer.py. Merging these two scripts intervenes storing intermediate cropped files in filesystems, which helps saving disk space.

import os
import sys
import os.path
import numpy as np
import skimage.io as io
import const_variables_list as CONST

from time import time
from io import BytesIO
from pyspark import SparkContext
from PIL import Image,ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_tiles_given_coordiantes(coordinates, image_path, image_data, output_path):

    cnn_input_size_x = CONST.alexnet_individual_image_size[0]
    cnn_input_size_y = CONST.alexnet_individual_image_size[1]

    if(hasattr(image_path,'filename')):
        prefix = str(image_path.filename.split("/")[-1])
        prefix = prefix.replace(".jpg","")
        # print(prefix)

    # Get the size of the image
    width, height = image_data.size

    # As PIL assumes coordinate (0,0) starts from Top left corner, we need to readjust the coordiantes for our convenience. As X-axis is same we dont need to change it.

    X = coordinates[0]
    Y = height - coordinates[1]

    print(X,Y)

    # Set the boundary of a fixed window which I will divide into tiles. The height and width of the bounded window needs to be a multiple of size 224, because
    # for AlexNet I am planning to use batches of images as inputs where each batch comprises of image having size of 224*224.

    max_multiple_of_224_X = int((width - X) / CONST.alexnet_individual_image_size[0])
    max_multiple_of_224_Y = int((height - Y) / CONST.alexnet_individual_image_size[1])

    # This max multiples can also be used to determine the number of tiles that I want to divide the cropped image into.
    number_of_tiles = (max_multiple_of_224_X, max_multiple_of_224_Y)

    # Check if the given output directory exists, if not create new directory.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    max_border_X = X + (CONST.alexnet_individual_image_size[0] * max_multiple_of_224_X)
    max_border_Y = Y + (CONST.alexnet_individual_image_size[1] * max_multiple_of_224_Y)

    # The commented outlines below creates the crop based on the boundary. I used it to varify if the tiles matches with its corresponding parts from the crop.

    four_points = (X, Y, max_border_X, max_border_Y)
    cropped_image = image_data.crop(four_points)
    cropped_image.save("cropped.jpg")

    # the outer for loop will iterate over the height.
    for i in range(0, number_of_tiles[1]):
        # the inner for loop will iterate over width.
        for j in range(0, number_of_tiles[0]):
            points = ((X+(j*cnn_input_size_x)), (Y+(i*cnn_input_size_y)),(X+((j+1)*cnn_input_size_x)),(Y+((i+1)*cnn_input_size_y)))
            frame = image_data.crop(points)
            frame.save(output_path + "/" +prefix + "_" + str(i) + "_" + str(j) + "_" + ".jpg")

def read_images(image_rawdata):
    return image_rawdata[0], np.array(io.imread(BytesIO(image_rawdata[1]), as_grey=True))


if __name__ == "__main__":

    application_start_time = time()

    """
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    job_name = sys.argv[4]
    coor = sys.argv[3]

    os.subprocess.call(["hadoop", "fs", "-rm", "-r", output_path])
    """
    input_path = "/discus/rashid/test_data/"
    output_path = "../image_tiles"
    job_name = "some_job"
    coor = "(140,490)"

    # Set spark configurations
    sc = SparkContext(appName=job_name)

    # When reading from local file system
    # images_rdd = sc.binaryFiles('file:///sparkdata/registration_images')

    # When reading from HDFS
    images_rdd = sc.binaryFiles(input_path)

    images_rdd = images_rdd.map(read_images) \
        .map(lambda image: make_tiles_given_coordiantes(coor, image[0], image[1], output_path))

    test_var = images_rdd.collect()

    application_end_time = time() - application_start_time

    sc.stop()

    print("------------------------------------------------")
    print(test_var)
    print("SUCCESS: Total time spent = {} seconds".format(round(application_end_time, 3)))
    print("------------------------------------------------")