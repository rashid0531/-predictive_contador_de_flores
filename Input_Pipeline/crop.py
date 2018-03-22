from PIL import Image,ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop_from_coordinates(coordinates,image):

    '''
    Given the coordinates, this function crops desired window from an image.
    :param:
        -coordinates: a tuple representing the coordinates.
        -image      : path to image.
    '''

    img = Image.open(image)

    # if(hasattr(img,'filename')):
    #     print(img.filename)
    #     save_as = 'cropped_'+img.filename+'.jpg'

    # Get the size of the image
    width, height = img.size

    # As PIL assumes coordinate (0,0) starts from Top left corner, we need to readjust the coordiantes for our convenience. As X-axis is same we dont need to change it.

    X = coor[0]
    Y= height-coor[1]

    # Set the resolution of a fixed window which I will crop from each image. The X axis and Y axis of the resolution needs to be a multiple of size 224, because
    # for AlexNet I am planning to use batches of images as inputs where each batch comprises of image having size of 224*224.

    max_multiple_of_224_X = int((width - X)/224)
    max_multiple_of_224_Y = int((height - Y) / 224)

    # This max multiples can also be used to determine the number of tiles that I want to divide the cropped image into.
    number_of_tiles = (max_multiple_of_224_X,max_multiple_of_224_Y)

    max_border_X  = X + (224*max_multiple_of_224_X)
    max_border_Y =  Y + (224*max_multiple_of_224_Y)

    four_points = (X,Y,max_border_X,max_border_Y)

    box = coor
    cropped_image = img.crop(four_points)
    cropped_image.save("dhiki.jpg")
    print(number_of_tiles)


if __name__== "__main__":

    coor = (140,490)
    crop_from_coordinates(coor,"./1120_frame000001.jpg")
