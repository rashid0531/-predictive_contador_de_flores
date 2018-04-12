import readData as read

from scipy import stats
import numpy as np
import math

def get_train_test_sets(label_file,train_ratio, binning = False):

    images, labels=read.process_label_files(label_file)

    filtered_imgs, filtered_labels = read.filter(images,labels)

    # Setting up Training set and Test set.
    trainset_ratio = train_ratio
    trainset_limit = int(len(filtered_imgs) * trainset_ratio)

    # Training set (image names)
    images_train = filtered_imgs[0:trainset_limit]

    # Testing set (image names)
    images_test = filtered_imgs[trainset_limit:]

    if binning == "True":

        filtered_labels_binned = list(map(get_bin,filtered_labels))
        # Training set (labels)
        labels_train = filtered_labels_binned[0:trainset_limit]
        # Testing set (labels)
        labels_test = filtered_labels_binned[trainset_limit:]

    else:
        labels_train = filtered_labels[0:trainset_limit]
        labels_test = filtered_labels[trainset_limit:]

    return images_train,labels_train,images_test,labels_test


def get_bin(value):

    '''
    This function returns bin number for a given value. The process of defining the bin is subjected to be changed over time.
    For the time being, we agreed to start with 4 classes: Bin0, Bin1, Bin2, Bin3.
    '''

    if (int(value) in range(56,82)):
        return 0

    elif (int(value) in range(82,107)):
        return 1

    elif (int(value) in range(107,220)):
        return 2

    # elif (int(value) in range(145,220)):
    #     return 3

