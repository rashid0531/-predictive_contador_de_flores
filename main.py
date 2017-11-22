import matplotlib.pyplot as plt

import input
import create_TFRecord


# Input file
path = "original.txt"

# Parameters for initializing dataset.
train_percent = 1
test_percent = 0.3

# Fetching training & testing set.
train_set, train_label, test_set, test_label = input.get_train_test_validation_sets(path=path,train_percent=train_percent,test_percent=test_percent)

# Read the image data as numpy array.
train_set = list(map(input.read_image_using_PIL,train_set))

# Creating TFRecords for training set.
tfRecord_name = 'train.tfrecords'
create_TFRecord.stash_example_protobuff_to_tfrecord(tfRecord_name,train_set,train_label)

# Ploting results received after transforming the tfrecord back to previous input format (numpy array)
returned_batched_images, returned_batched_lables= create_TFRecord.read_tfrecords_as_batch(tfRecord_name,3)
print(returned_batched_lables)

show_images = True
if (show_images):
    for i in range(3):
        plt.imshow(returned_batched_images[i,:,:,:])
        plt.show()
