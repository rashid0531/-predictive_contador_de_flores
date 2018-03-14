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
    box = coor
    cropped_image = img.crop(box)
    cropped_image.save('cropped_image.jpg')


coor = (200, 200, 1108, 654)
crop_from_coordinates(coor,"../frame000001.jpg")
