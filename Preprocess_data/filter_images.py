import glob
import sys
import os

def get_files(directory,coordinate_path):

    cameraid_capturing_days=[]
    for filename in glob.glob(directory+"/*"):
        cameraid_capturing_days.append(filename.split("/")[-1])

    print(len(cameraid_capturing_days))

    # Check if the file exists, if not then create a new file.
    try:
        with open(coordinate_path,'r') as file_obj:
            file_contents = file_obj.read()

    except FileNotFoundError:
        msg = coordinate_path + " does not exist."
        print(msg)
        prompt = input("Want to create the co-ordinate file [y/n]? \n")

        # After creating the file I want to populate it with the camera days. So that it becomes easier for me to manually keep record of coordinates for each camera.
        # This part of code automates boring stuffs.

        if (prompt == 'y' or "yes"):
            with open(coordinate_path,'w') as file_obj:
                for i in cameraid_capturing_days:
                    file_obj.write(i+" : "+"\n")


if __name__ == "__main__":

    coordinate_path = "/discus/P2IRC/rashid/co_ordinates.txt"

    get_files(sys.argv[1],coordinate_path)
