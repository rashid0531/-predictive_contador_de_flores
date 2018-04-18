import glob
import sys
import info
sys.path.append(info.path_to_input_pipeline_discus)
# sys.path.append(info.path_to_input_pipeline_local)
import readData as read
import statistical_summary as stats
import numpy as np
import itertools

def get_counts_for_each_camera(directory):

    cameraid_capturing_days=[]
    for filename in glob.glob(directory+"/*"):
        cameraid_capturing_days.append(filename)

    cameraid_capturing_days = sorted(cameraid_capturing_days)

    # Not using days when flower count is almost zero.
    cameraid_capturing_days = cameraid_capturing_days[5:-5]

    all_image_path=[]
    all_labels = []
    for each_entry in cameraid_capturing_days:

        # Formating eachline
        characters_needTobeRemoved = ["[","'","]"]
        label_file_path_for_each_day = str(glob.glob(each_entry+"/*"))

        for i in range(0,len(characters_needTobeRemoved)):
            label_file_path_for_each_day = label_file_path_for_each_day.replace(characters_needTobeRemoved[i],"")

        img,label = read.process_label_files(label_file_path_for_each_day)
        all_image_path.append(img)
        all_labels.append(label)

    return all_image_path,all_labels

if __name__== "__main__":

    default_path = "/u1/rashid/FlowerCounter_Dataset_labels"
    all_img_path,all_labels = get_counts_for_each_camera(default_path)

    flatten_labels = list(itertools.chain.from_iterable(all_labels))

    sorted_flatten_labels = sorted(flatten_labels)

    # stats.make_plot(sorted_flatten_labels, semilog_y = True)

    filtered_list = list(filter(lambda x: x > 0, sorted_flatten_labels))

    # stats.make_histogram(filtered_list,236)

    median,mean,stdev = np.median(filtered_list),np.mean(filtered_list),np.std(filtered_list)

    print("Median : {} , Mean : {} Standard Deviation : {} ".format(median,mean,stdev))

