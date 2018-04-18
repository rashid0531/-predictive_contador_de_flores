import numpy as np
from PIL import Image,ImageFile
import tensorflow as tf
import random
import Data_Visualization.statistical_summary as stats
import math
import prepare_dataset as prepare

def process_label_files(label_file):
    """
    @:param:

    :return:
    """
    # To store image and its label as pair. I used it when the data were converted to tfrecord.
    # paired_img_label = []

    listOf_image_paths = []
    labels = []

    # For local repository add the following prefix.

    # input_prefix = "/u1/rashid/FlowerCounter_Dataset/"

    input_prefix = "/home/rashid/Projects/FlowerCounter/dataset/"

    '''
    Each line in the text file is saved as " u'hdfs://discus-p2irc-master:54310/user/hduser/rashid/output/1109-0710/frame001117_0_3.jpg' "
    '''

    try:
        with open(label_file,'r') as file_obj:

            file_contents = file_obj.readlines()
            # Randomized lines inside file.
            # random.shuffle(file_contents)

            for each_line in file_contents:

                # stripping the new lines ('\n') from each line
                each_line = each_line.strip()
                # filtering '(' and ')' from each line.
                each_line = each_line.replace('(','')
                each_line = each_line.replace(')','')
                image_path,label = each_line.split(',')

                # The next two lines are subjected to be changed based on the location of data. For HDFS it needs to be changed.
                image_path = input_prefix + ('/'.join(image_path.split('/')[-2:]))
                # This line is to replace the last ' in the imagepath.
                image_path = image_path.replace("'",'')

                listOf_image_paths.append(image_path)
                labels.append(int(label))

                # paired_img_label.append((image_path,label))
    except FileNotFoundError:
        msg = label_file + " does not exist."
        print(msg)

    return listOf_image_paths, labels

def read_image_using_PIL(img_label_pair):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(img_label_pair[0])
    img = np.asarray(img, np.uint8)

    # To check if the image load is successful.
    # img = Image.fromarray(img, 'RGB')
    # img.show()

    return (img.tobytes(),img_label_pair[1])

# This function was taken from Tensorflows documentation.
# Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
def _parse_function(filename, label, onehot = False, number_of_classes = False):

  image_string = tf.read_file(filename)

  image_decoded = tf.image.decode_jpeg(image_string,channels=3)
  image_resized = tf.image.resize_images(image_decoded, [224, 224])
  image = tf.cast(image_resized, tf.float32)

  if (onehot == True):
      # convert the label to one-hot encoding
      onehot_encoded_label = tf.one_hot(label, number_of_classes)

      return image, onehot_encoded_label

  else:
        return image, label


def filter(images,labels):

    range_min,range_max= np.median(labels) - np.std(labels), np.median(labels) + np.std(labels)

    range_max += 29

    filtered_image=[]
    filtered_labels= []

    for i in range(len(labels)):
        if (labels[i] < math.ceil(range_max) and labels[i] > math.floor(range_min)):
            filtered_image.append(images[i])
            filtered_labels.append(labels[i])

    return filtered_image,filtered_labels


if __name__ == "__main__":

    input_path_local = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"
    input_path_local_snd = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0711/part-00000"
    # input_path_local = "/home/rashid/Projects/FlowerCounter/label/part-00000"
    img,labels = process_label_files(input_path_local)
    img_2,labels_2 = process_label_files(input_path_local_snd)

    stats.make_histogram_twoset(labels,labels_2,200)
    stats.make_cdf_twoset(labels, labels_2, 200)
