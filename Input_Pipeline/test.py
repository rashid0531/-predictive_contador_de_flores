import tensorflow as tf
import glob
import Input_Pipeline.readData as read


label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/1109-0710/part-00000"
images,labels = read.process_label_files(label_input_path)






