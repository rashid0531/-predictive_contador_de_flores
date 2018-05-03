import numpy as np
import skimage.io as io
import cv2

from PIL import Image,ImageFile

def read_image_using_PIL(img):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(img)
    img = np.asarray(img, np.uint8)
    return img

def format_coordinates(label_file):

    input_prefix = "/u1/rashid/FlowerCounter_Dataset/"

    coordinates_forAllImages =[]
    imagePath_forAllImages = []

    try:
        with open(label_file, 'r') as file_obj:

            file_contents = file_obj.readlines()
            # Randomized lines inside file.
            # random.shuffle(file_contents)
            for each_line in file_contents:

                each_line = each_line.strip()
                image_path,coordinates = each_line.split(',',1)[0],each_line.split(',',1)[1]
                # print(image_path)
                # filtering '(' and ')' from each line.
                image_path = image_path.replace('(', '')
                image_path = image_path.replace("'", '')
                image_path = input_prefix + ('/'.join(image_path.split('/')[-2:]))
                # This line is to replace the last ' in the imagepath.
                image_path = image_path.replace("'", '')
                imagePath_forAllImages.append(image_path)

                coordinates = coordinates[:-1]
                coordinates = eval(coordinates)
                coordinates_forAllImages.append(coordinates)

    except FileNotFoundError:
        msg = label_file + " does not exist."
        print(msg)

    return imagePath_forAllImages,coordinates_forAllImages


if __name__== '__main__':

    label_file = "/u1/rashid/Sample_output/1109-0709/part-00000"
    imagePaths, Coordinates = format_coordinates(label_file)

    dens_im = read_image_using_PIL(imagePaths[4])

    for i in Coordinates[4]:
        cv2.circle(dens_im, center =(int(i[0]), int(i[1])), radius=1, color=(255,48,48), thickness=2)

    io.imshow(dens_im)
    # io.imshow(dens_im)
    io.show()


