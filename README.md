# Capstone -> Karyogram generatot


# About: Model to produce an end to end karyogram in the correct orientation

## step 1: 
Get seperated photos of each class either by running utils\seperate_class.py or downloading it from the shared drive. 

## step 2: 
Add the bash script, yaml file and the approrpaite image file for the class into sockeye so where the paths are indicated in yaml/bash and submit a job to run YOLO 
NOTE: before you run you can run the python lines in your command line to make sure there are no compiler error before you submit your job to sockeye

## step 3: 
Once  you have the output images of yolo boxes and their labels run the utils\yolo_to_mask.py and subsequently the utils\generate_masked_photos.py

## step 4: 
Now that you have images of  a specific chromosome with a binary  form of  only including the masked part and whiting out the surroundings  input it 
into the  CVAE model for training/testing 

## step 5:
Analyzing the latent space from cvae