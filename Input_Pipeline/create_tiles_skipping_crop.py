'''
Author: Mohammed Rashid Chowdhury
email:  <mrc689@mail.usask.ca>
Script name: create_tiles_skipping_crop.py
'''

# This script merges crop.py and slicer.py. Merging these two scripts intervenes storing intermediate cropped files in filesystems, which helps saving disk space.

from PIL import Image,ImageFile
import os
import const_variables_list as CONST

ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_tiles_given_coordiantes(coordinates,input_path,output_directory):

    img = Image.open(input_path)

    cnn_input_size_x = CONST.alexnet_individual_image_size[0]
    cnn_input_size_y = CONST.alexnet_individual_image_size[1]

    if(hasattr(img,'filename')):
        prefix = str(img.filename.split("/")[-1])
        prefix = prefix.replace(".jpg","")
        # print(prefix)

    # Get the size of the image
    width, height = img.size

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
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    max_border_X = X + (CONST.alexnet_individual_image_size[0] * max_multiple_of_224_X)
    max_border_Y = Y + (CONST.alexnet_individual_image_size[1] * max_multiple_of_224_Y)

    # The commented outlines below creates the crop based on the boundary. I used it to varify if the tiles matches with its corresponding parts from the crop.

    four_points = (X, Y, max_border_X, max_border_Y)
    cropped_image = img.crop(four_points)
    cropped_image.save("cropped.jpg")

    # the outer for loop will iterate over the height.
    for i in range(0, number_of_tiles[1]):
        # the inner for loop will iterate over width.
        for j in range(0, number_of_tiles[0]):
            points = ((X+(j*cnn_input_size_x)), (Y+(i*cnn_input_size_y)),(X+((j+1)*cnn_input_size_x)),(Y+((i+1)*cnn_input_size_y)))
            frame = img.crop(points)
            frame.save(output_directory + "/" +prefix + "_" + str(i) + "_" + str(j) + "_" + ".jpg")


if __name__== "__main__":


    coor = (10,700)
    make_tiles_given_coordiantes(coor,"../Test_dataset/1109-0710-frame000301.jpg","../Test_dataset/image_tiles")