'''
Author: Mohammed Rashid Chowdhury
email:  <mrc689@mail.usask.ca>
Script name: another_slicer.py
'''

# I implemented the slicing functionality from scratch because the library that i was using in slicer.py (image_slicer) doesn't work as expected.

import os
from PIL import Image,ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

output_directory = "/u1/rashid/Data/predictive_flower_counter/splited_images"
input_directory = "/discus/"

# This function slices a given image into user given number of tiles.
def make_Tiles(img_path,number_of_tiles_tuple,directory):
    img = Image.open(img_path)
    # print(img.size)

    # Assuming each input image which will be fed to Alexnet has an dimension of 224*224
    input_image_size = (224, 224)

    # Getting image name
    # if(hasattr(img,'filename')):
    #     print(img.filename)
    #     save_as = 'cropped_'+img.filename+'.jpg'

    # Check if the given directory exists, if not create new directory.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the size of the image
    width, height = img.size

    # the outer for loop will iterate over the height.
    for i in range(0,number_of_tiles_tuple[1]):
        # the inner for loop will iterate over width.
        for j in range(0,number_of_tiles_tuple[0]):
            points=(j*input_image_size[0],i*input_image_size[1],(j+1)*input_image_size[0],(i+1)*input_image_size[1])
            frame = img.crop(points)
            frame.save(directory+"/crop_"+str(i)+"_"+str(j)+"_"+".jpg")


if __name__ == "__main__":
    # test if the function works.
    make_Tiles('../Preprocess_data/dhiki.jpg',(5,2),"../image_tiles")




