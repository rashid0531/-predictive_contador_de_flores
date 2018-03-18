import glob
import sys

def get_files(directory):

    dict={}

    for filename in glob.glob(directory+"/*"):
        dict['collection_date'] = filename.split("/")[-1]
        # print(dict['collection_date'])
        camera_id=[]
        for sub_filename in glob.glob(filename+"/*"):

            camera_info = []

            camera_info.append(sub_filename.split("/")[-1])
        # dict['camera_id'] = camera_id
            for sub_file_images in glob.glob(sub_filename+"/*"):

                days =[]
                days.append(sub_file_images.split("/")[-1])

            print(days)


if __name__ == "__main__":

    get_files(sys.argv[1])
