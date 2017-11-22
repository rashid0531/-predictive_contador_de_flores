a=[1,2,3]
b=[4,5,6]
z=10
c = list(map(lambda x,y: x+y,a,b))

# print(c)

def get_bin(value):

    '''
    This function returns bin number for a given value. The process of defining the bin is subjected to be changed over time.
    For the time being, we agreed to start with 5 classes: Bin0, Bin1, Bin2, Bin3, Bin4.
    '''

    if (int(value) in range(0,700)):
        return 0

    elif (int(value) in range(700,800)):
        return 1

    elif (int(value) in range(800,900)):
        return 2

    elif (int(value) in range(900,1000)):
        return 3

    else:
        return 4

def list_imageData_with_labels(directory):

    '''
    This function creates label for each image files for the given directory.

    :param:
    -directory: The path where our data is saved.

    :return:
    -file_name: a list that contains the image names.
    -labels: a list of labels for correspondent image file

    '''
    labels =[]
    file_name = []
    #path = "../test_set/flower_plots/"

    # for testing small dataset
    path = "flower_plots/"

    with open(directory) as file_object:

        for each_line in file_object:
            image_name = str(path+each_line.split( )[0])
            count = each_line.split( )[1]
            file_name.append(image_name)
            labels.append(get_bin(count))

    return file_name,labels

files,labels = list_imageData_with_labels("original.txt")

print(len(files))
print(files)

limit = int(len(files)*0.7)
print(limit)
train_set = files[0:limit]

test_set = files[limit:]

print(train_set)
print(test_set)

