# #files,labels = list_imageData_with_labels("../test_set/original.txt")
#
# # For small dataset to test in a local machine
# files,labels = list_imageData_with_labels("original.txt")
# files = list(map(read_image_using_PIL,files))
# tfRecord_name = 'train.tfrecords'
#
# create_TFRecord.stash_example_protobuff_to_tfrecord(tfRecord_name,files,labels)
#
# # Ploting results received after transforming the tfrecord back to previous input format (numpy array)
# returned_batched_images, returned_batched_lables= create_TFRecord.read_tfrecords_as_batch(tfRecord_name,3)
# print(returned_batched_lables)
#
# show_images = True
# if (show_images):
#     for i in range(3):
#         plt.imshow(returned_batched_images[i,:,:,:])
#         plt.show()


# Ploting results received after transforming the tfrecord back to previous input format (numpy array)
# returned_batched_images, returned_batched_lables= create_TFRecord.read_tfrecords_as_batch(tfRecord_name,5)

# print(returned_batched_lables)
#
# show_images = True
# if (show_images):
#     for i in range(3):
#         plt.imshow(returned_batched_images[i,:,:,:])
#         plt.show()


'''
Script name: create_tiles_skipping_crop.py


    # The commented outlines below creates the crop based on the boundary. I used it to varify if the tiles matches with its corresponding parts from the crop.

    four_points = (X, Y, max_border_X, max_border_Y)
    cropped_image = img.crop(four_points)
    cropped_image.save("cropped.jpg")


    # The following commented out nested for loop is easier to understand, how I am cropping the tiles, but it needs slightly more works for naming convention.

    counter_index_x, counter_index_y = 0, 0

    # the outer for loop will iterate over the height.
    for i in range(Y,max_border_Y,CONST.alexnet_individual_image_size[1]):

        counter_index_x = 0
        # the inner for loop will iterate over width.
        for j in range(X,max_border_X,CONST.alexnet_individual_image_size[0]):
            points=(j,i,j+CONST.alexnet_individual_image_size[0],i+CONST.alexnet_individual_image_size[1])
            print(points)
            frame = img.crop(points)
            frame.save(output_directory+"/crop_"+str(counter_index_y)+"_"+str(counter_index_x)+"_"+".jpg")
            counter_index_x += 1

        counter_index_y += 1

'''