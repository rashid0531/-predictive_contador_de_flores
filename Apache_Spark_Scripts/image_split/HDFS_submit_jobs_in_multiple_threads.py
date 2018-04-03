'''
Author: Mohammed Rashid Chowdhury,Habib Sabiu
email:  <mrc689@mail.usask.ca>
email:
Script name: HDFS_submit_jobs_in_multiple_threads.py
'''

# This script is the Apache Spark's version of creating tiles. This script aims to reduce the total elapsed time of slicing cropped images into small tiles.

import os
import glob
import os.path
import threading
import subprocess
import numpy as np
import skimage.io as io
import const_variables_list as CONST

from io import BytesIO
from time import time, sleep
from PIL import Image,ImageFile
from pyspark import SparkContext


#def make_tiles_given_coordiantes(coordinates, image_path, image_data, output_path):
def make_tiles_given_coordiantes(data):

    coordinates = data[0]
    image_path = data[1]
    image_data =  data[2]
    output_path = data[3]

    cnn_input_size_x = CONST.alexnet_individual_image_size[0]
    cnn_input_size_y = CONST.alexnet_individual_image_size[1]

    prefix = image_path.split("/")[-1]
    prefix = prefix.replace(".jpg","")

    # Get the size of the image
    width, height = len(image_data[0]),len(image_data)

    # As PIL assumes coordinate (0,0) starts from Top left corner, we need to readjust the coordiantes for our convenience. As X-axis is same we dont need to change it.

    X = coordinates[0]
    Y = height - coordinates[1]

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

    # four_points = (X, Y, max_border_X, max_border_Y)
    # cropped_image = image_data.crop(four_points)
    # cropped_image.save("cropped.jpg")

    img = Image.fromarray(image_data, 'RGB')

    # the outer for loop will iterate over the height.
    for i in range(0, number_of_tiles[1]):
        # the inner for loop will iterate over width.
        for j in range(0, number_of_tiles[0]):
            points = ((X+(j*cnn_input_size_x)), (Y+(i*cnn_input_size_y)),(X+((j+1)*cnn_input_size_x)),(Y+((i+1)*cnn_input_size_y)))
            frame = img.crop(points)
            frame.save(output_path + prefix + "_" + str(i) + "_" + str(j) + ".jpg")


def read_images(image_rawdata):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return image_rawdata[0], np.array(io.imread(BytesIO(image_rawdata[1])))

def run_jobs_in_parallel(full_images_input_path, full_images_output_path, coor, sc):

    images_rdd = sc.binaryFiles(full_images_input_path) \
        .map(read_images) \
        .map(lambda  image: (coor, image[0], image[1], full_images_output_path)) \
        .foreach(make_tiles_given_coordiantes)


if __name__== "__main__":

    input_path = "/data/mounted_hdfs_path/user/hduser/cameradays/"
    co_ordinate_path = "/data/mounted_hdfs_path/user/hduser/rashid/co_ordinates.txt"
    output_path = "/data/mounted_hdfs_path/user/hduser/rashid/output/"

    # input_path = "/discus/P2IRC/Summer2016/image-data/timelapse_images/cameradays/"
    # co_ordinate_path = "/discus/P2IRC/rashid/co_ordinates.txt"
    # output_path = "../output/"

    if not os.path.exists(input_path):
       print("Input path doesn't exit.")

    if not os.path.exists(co_ordinate_path):
       print("Co_ordinate file doesn't exit.")

    # To store the folder names (which is basically the cameraid followed by the capturing day).
    cameraid_capturingdays=[]

    for filename in glob.glob(input_path+"*"):

        cameraid_capturingdays.append(filename.split("/")[-1])

    # For testing I am only considering camera_id:1109

    filter_id = '1109'

    cameraid_capturingdays_partial = list(filter(lambda x:x.split("-")[0] == filter_id,cameraid_capturingdays))

    cameraid_capturingdays_partial = sorted(cameraid_capturingdays_partial)

    co_ordinates_partial = []

    # open the coordinate file
    with open(co_ordinate_path) as file_object:

        all_lines =file_object.readlines()

        # save the single Co-ordinate of each images captured by camera id 1108
        # checks each line of file object and filters by 1108

        for line in all_lines:
            if (((line.strip().split(":")[0]).split("-")[0])) == filter_id:
                co_ordinates_partial.append(line.strip())

    application_start_time = time()

    sc = SparkContext(appName= "ImageSplitting")

    all_threads = []

    for i in range(0,len(cameraid_capturingdays_partial)):

        coor = co_ordinates_partial[i].split(":")[-1]
        coor = coor.split(",")

        coor = (int(coor[0]),int(coor[1]))

        hdfs_output_path = "hdfs://discus-p2irc-master:54310/user/hduser/rashid/output/"+cameraid_capturingdays_partial[i]+"/"

        # Delete the hdfs_output_path if it already exists in hdfs
        subprocess.call(["hadoop", "fs", "-rm", "-r", hdfs_output_path])

        full_images_input_path = "file://"+input_path+cameraid_capturingdays_partial[i]+"/"
        full_images_output_path = output_path+cameraid_capturingdays_partial[i]+"/"

        t = threading.Thread(target=run_jobs_in_parallel, args=(full_images_input_path, full_images_output_path, coor, sc))
        all_threads.append(t)


        t.start()

        print("--------------------------")
        print("Submitted job number: " + str(i))
        print("--------------------------")

    [t.join() for t in all_threads]

    sc.stop()

    application_end_time = time() - application_start_time

    print("------------------------------------------------")
    print("SUCCESS: Total time spent = {} seconds".format(round(application_end_time, 3)))
    print("------------------------------------------------")