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