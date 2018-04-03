import tensorflow as tf
import glob
import os


images_input_path = "/u1/rashid/FlowerCounter_Dataset/"
label_input_path = "/u1/rashid/FlowerCounter_Dataset_labels/"

for file in glob.glob(label_input_path+"*"):
    for ascii_txt in glob.glob(file+"/*"):
        remove_file = ascii_txt.split("/")[-1]
        if (remove_file == "_SUCCESS"):
            os.remove(ascii_txt)