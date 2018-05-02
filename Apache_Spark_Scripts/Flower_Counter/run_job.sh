!#bin/bash

#spark-submit --master spark://discus-p2irc-master:7077 --py-files canola_timelapse_image.py imageFlowerCounterNew.py \
#                hdfs://discus-p2irc-master:54310/user/hduser/habib/still_camera_images/ \
#                hdfs://discus-p2irc-master:54310/user/hduser/habib/flower_counter_output/ imageFlowerCounter


spark-submit --master local[20] --py-files canola_timelapse_image.py getCoordinates_distributed.py /u1/rashid/Sample_Dataset /u1/rashid/Sample_output/ imageFlowerCounter


#spark-submit --master local[20] --py-files canola_timelapse_image.py FlowerCounter_Distributed.py ../../Test_dataset/image_tiles/ ../../Test_dataset/output/count/ imageFlowerCounter
