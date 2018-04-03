'''
This script is for converting and collating the small images into a big tfrecord file.
'''

import tensorflow as tf
import numpy as np
import Input_Pipeline.readData as read

def create_TFRecord(file,name):

    img_label = read.process_label_files(file)
    raw_img_label = list(map(read.read_image_using_PIL,img_label))

    #convert each pair of image and its label to example protocall buffer.
    proto_buffs = list(map(transfer_to_protobuff, raw_img_label))

    tfrecord_name = name
    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    for example in proto_buffs:
        writer.write(example.SerializeToString())

    writer.close()


def transfer_to_protobuff(raw_pair_of_img_lable):
    '''
    This function writes the image data to a proto buffer.
    :param:
    -raw_pair_of_img_lable[0]  : raw Image data.
    -raw_pair_of_img_lable[1]  : Image label.
    :return:
    an example proto buffer.
    '''
    feature = {
        'imagedata': _bytes_feature(raw_pair_of_img_lable[0]),
        'label': _int64_feature(int(raw_pair_of_img_lable[1]))
    }

    # creating example proto buffer
    example= tf.train.Example(features= tf.train.Features(feature = feature))

    return example


def stash_example_protobuff_to_tfrecord(name,files,labels):
    '''
    This function converts whatever data we provide into a tensorflow supported format called tfrecord.
    :param:
    -name   : name of the tfrecord.
    -files  : A list of image filenames.
    -labels : A list containing labels for each imagefile.
    '''
    tfrecord_name = name
    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    proto_buffs = list(map(transfer_to_protobuff,files,labels))

    for example in proto_buffs:
        writer.write(example.SerializeToString())

    writer.close()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":

    input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"
    create_TFRecord(input_path,"Allah.tfrecords")
