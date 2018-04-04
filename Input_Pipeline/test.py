import tensorflow as tf
import glob
import Input_Pipeline.readData as read

label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"
images,labels = read.process_label_files(label_input_path)

# A vector of filenames.
images_input = tf.constant(images)
images_labels = tf.constant(labels)

dataset = tf.data.Dataset.from_tensor_slices((images_input, images_labels))
dataset = dataset.map(read._parse_function)
#
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
    elem = sess.run(next_element)
    print(elem)
    elem = sess.run(next_element)
    print(elem)