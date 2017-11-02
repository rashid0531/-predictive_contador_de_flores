import tensorflow as tf

def transfer_to_protobuff(image,label):
    '''
    This function writes the image data to a proto buffer.
    :param:
    -image  : raw Image data.
    -label  : Image label.
    :return:
    an example proto buffer.
    '''
    feature = {
        'imagedata': _bytes_feature(image),
        'label': _int64_feature(label)
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
    



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))