import readData as read
import glob
import math
import numpy as np
import sys
import itertools
import random
import info
# sys.path.append(info.path_to_Data_Visualization_discus)
sys.path.append(info.path_to_Data_Visualization_local)

import statistical_summary as stats

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

def get_last_index(lst,item):
    try:
        idx = len(lst) - lst[::-1].index(item) - 1

    except ValueError:
        idx = -1

    return idx


def get_left_right(sorted_arr,i,histogram_interval):
    # set the left as the min of the selected bin

    try:
        left = sorted_arr.index(i)
    except ValueError:
        left = -1

    if (left == -1):
        counter = i + 1

    while (left == -1):
        left = sorted_arr.index(counter)
        counter += 1

    # set the left as the min of the selected bin

    right = get_last_index(sorted_arr, i + histogram_interval)

    if (right == -1):
        counter_r = (i + histogram_interval) - 1

    while (right == -1):
        right = get_last_index(sorted_arr, counter_r)
        counter_r -= 1

    return left,right

def prepare(label_folder):

    cameraid_capturing_days = []
    for filename in glob.glob(label_folder + "/*"):
        cameraid_capturing_days.append(filename)

    cameraid_capturing_days = sorted(cameraid_capturing_days)

    # Not using days when flower count is almost zero.
    cameraid_capturing_days = cameraid_capturing_days[5:-5]

    all_image_path = []
    all_labels = []
    for each_entry in cameraid_capturing_days:

        # Formating eachline
        characters_needTobeRemoved = ["[", "'", "]"]
        label_file_path_for_each_day = str(glob.glob(each_entry + "/*"))

        for i in range(0, len(characters_needTobeRemoved)):
            label_file_path_for_each_day = label_file_path_for_each_day.replace(characters_needTobeRemoved[i], "")

        img, label = read.process_label_files(label_file_path_for_each_day)
        all_image_path.append(img)
        all_labels.append(label)

    return all_image_path, all_labels

if __name__== "__main__":

    # default_path = "/u1/rashid/FlowerCounter_Dataset_labels"
    default_path = "/home/rashid/Projects/FlowerCounter/label_dataset"

    all_img_path,all_labels = prepare(default_path)
    all_img_path,all_labels = np.array(all_img_path),np.array(all_labels)

    flatten_img_path = list(itertools.chain.from_iterable(all_img_path))
    flatten_labels = list(itertools.chain.from_iterable(all_labels))

    filtered_img,filtered_labels = read.filter(flatten_img_path,flatten_labels)

    paired_label_img = list(zip(filtered_labels,filtered_img))
    paired_label_img.sort()
    sorted_label,sorted_img = zip(*paired_label_img)
    # print(sorted_img[7000],sorted_label[7000])

    stats.CountFrequency(sorted_label)

    number_of_bins = 10
    # stats.make_histogram(sorted(filtered_labels), number_of_bins)

    min, max = sorted(filtered_labels)[0], sorted(filtered_labels)[-1]
    histogram_interval = int((max - min) / number_of_bins)

    print(sorted_label[15439])
    # print(len(sorted_label))

    frequency_inside_bins = []

    for i in range(min,max,histogram_interval+1):

        left, right = get_left_right(sorted_label,i,histogram_interval)

        # print(left,right)

        frequency_inside_bins.append(right-left)

    print(frequency_inside_bins)

    stats.make_histogram(sorted(filtered_labels), number_of_bins)

    # print(len(filtered_img),len(filtered_labels))

    # print(filtered_img[-13],filtered_labels[-13])

    # paired_img_label = list(zip(filtered_img, filtered_labels))
    # #
    # # for i in range (0,5):
    # #     random.shuffle(paired_img_label)
    # #
    # filtered_img_shuffled, filtered_labels_shuffled = zip(*paired_img_label)
    #
    # stats.make_histogram(sorted(filtered_labels_shuffled),number_of_bins = 10)




