import Input_Pipeline.readData as read
# import readData as read

from scipy import stats
import numpy as np
import math

def get_train_test_sets(label_file,train_ratio):

    images, labels=read.process_label_files(label_file)

    filtered_imgs, filtered_labels = read.filter(images,labels)

    # Setting up Training set and Test set.
    trainset_ratio = train_ratio
    trainset_limit = int(len(filtered_imgs) * trainset_ratio)

    # Training set
    images_train = filtered_imgs[0:trainset_limit]
    labels_train = filtered_labels[0:trainset_limit]

    # Testing set
    images_test = filtered_imgs[trainset_limit:]
    labels_test = filtered_labels[trainset_limit:]

    return images_train,labels_train,images_test,labels_test


def get_bin(value):

    '''
    This function returns bin number for a given value. The process of defining the bin is subjected to be changed over time.
    For the time being, we agreed to start with 4 classes: Bin0, Bin1, Bin2, Bin3.
    '''

    if (int(value) in range(0,40)):
        return 0

    elif (int(value) in range(40,70)):
        return 1

    elif (int(value) in range(70,145)):
        return 2

    elif (int(value) in range(145,220)):
        return 3


def filter(images,labels):

    range_min,range_max= np.median(labels) - np.std(labels), np.median(labels) + np.std(labels)

    filtered_image=[]
    filtered_labels= []

    for i in range(len(labels)):
        if (labels[i] < math.ceil(range_max) and labels[i] > math.floor(range_min)):
            filtered_image.append(images[i])
            filtered_labels.append(labels[i])

    return filtered_image,filtered_labels

