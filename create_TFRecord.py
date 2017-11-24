import tensorflow as tf
import numpy as np

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

    proto_buffs = list(map(transfer_to_protobuff,files,labels))

    for example in proto_buffs:
        writer.write(example.SerializeToString())

    writer.close()


def read_singleExample_tfrecord(tfrecord_name):
    '''
    This function converts the given tfrecord file back to the original format (numpy array or string)
    :param:
    -tfrecord_name : name of TFRecord file.

    '''
    #Create a queue to hold files
    filename_queue = tf.train.string_input_producer([tfrecord_name], num_epochs=1)

    # A TFRecord reader to read each element of the file queue.
    reader = tf.TFRecordReader()

    # A feature dictionary which will hold the extracted features from deserialized protobuff.
    feature = {
        'imagedata': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    # Get the serialized example protobuff from tfrecord
    _, serialized_example_proto = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example_proto, features=feature)

    # Reconstruct the image from 'imagedata' stored in features dictionary.
    img = tf.decode_raw(features['imagedata'],tf.uint8)

    # Reshape image data into the original shape
    img = tf.reshape(img, [227,227, 3])

    # Reconstruct the labels
    labels = tf.cast(features['label'], tf.int32)

    # enable to see the visual plotting of single image
    show = False

    if(show):

        # Dont know why i needed these following lines.
        sess = tf.InteractiveSession()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img = sess.run(img)
        img = img.astype(np.uint8)
        sess.close()

    return img , labels

def read_tfrecords_as_batch(tfrecord_name,batch_size):
    '''
    This function runs "read_singleExample_tfrecord" as batch.

    :param tfrecord_name:
    :param batch_size: size of the batch.
    :return: batch of images converted back to its original format. (array)
    '''
    image,label = read_singleExample_tfrecord(tfrecord_name)

    '''
    For small dataset, one better practice is to use tf.train.batch instead of tf.train.shuffle_batch as 
    the later enqueues each image and label multiple times in the tf.RandomShuffleQueue.
    This explanation mentioned above is taken from: https://stackoverflow.com/questions/39283605/regarding-the-use-of-tf-train-shuffle-batch-to-create-batches 
    '''
    # To simply avoid queueing errors set allow_smaller_final_batch to 'True'
    image_batches, label_batches = tf.train.batch([image, label], batch_size = batch_size, num_threads=4, allow_smaller_final_batch=True)

    # For large dataset comment out the following line.
    #image_batches,label_batches = tf.train.shuffle_batch([image,label],batch_size=3,num_threads=4,capacity=3,min_after_dequeue=1)


    # Dont know why i needed these following lines.
    sess = tf.InteractiveSession()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    batched_images,batched_labels = sess.run([image_batches,label_batches])

    return batched_images,batched_labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _bytes_feature(value):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))