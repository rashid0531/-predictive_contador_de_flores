import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
import skimage.io as io

def createDensityMap(arr, points):

    # creates an array of only zero values based on the shape of given array.
    dens_map = np.zeros(shape=[arr.shape[0], arr.shape[1]])

    # Take each coordinates from the list "points" and put 255 on that position.
    for point in points:
        dens_map[point[1]][point[0]] = 255

    # Normalizing the array so that the sum over the whole array equals to the number of elements in the list "points".
    normalized = (dens_map - np.min(dens_map)) / (np.max(dens_map) - np.min(dens_map))
    sigmadots = 7
    dot_anno = gaussian_filter(normalized, sigmadots)
    dot_anno.astype(np.float32)

    return dot_anno

def getAnnotated_points(image):

    # zeros = np.zeros((100, 100), dtype=np.uint8)
    # zeros[:5, :5] = 255

    indices = np.where(image == [255])
    for i in range(0,len(indices)):
        print((indices[1][i],indices[0][i]))

def read_image_using_PIL(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(image)
    image = image.resize((227,227))
    image = np.asarray(image, np.uint8)

    if(image.flags['WRITEABLE'] == False):
        image.setflags(write=1)

    return image


if __name__ == "__main__":

    array = np.zeros(shape=[224, 224])
    points = [(185, 220), (97, 219), (80, 219), (0, 218), (13, 211), (25, 208), (0, 207), (192, 205), (178, 199), (194, 190), (180, 187), (172, 129), (42, 127), (59, 126), (50, 118), (169, 117), (34, 117), (57, 112), (68, 111), (42, 110), (164, 108), (141, 108), (129, 105), (48, 99), (97, 97), (60, 97), (2, 97), (75, 94), (144, 91), (133, 91), (59, 83), (0, 82), (72, 75), (86, 74), (102, 73), (205, 67), (182, 66), (93, 65), (80, 63), (101, 60), (174, 56), (88, 55), (185, 48), (165, 25), (177, 21), (170, 21), (175, 9), (0, 6), (182, 0), (168, 0), (11, 0)]

    # plt.imshow(createDensityMap(array,points=points))
    # plt.show()

    img = read_image_using_PIL("/u1/rashid/FlowerCounter/predictive_contador_de_flores/Preprocess_data/square_circle_opencv.jpg")
    getAnnotated_points(img)

    io.imshow(img)
    io.show()

