# Activities of Daily Living - [First Person Activity Recognition Independent Study Texas State University] 

### Overview
This repository is a compilation of programs to do network training and video processing, particularly on the ADL (Activites of Daily Living) dataset from 2012 to work with the Kinetics-i3D model. The ADL dataset consists of 1 million frames of unscripted morning activity in the 1st person point of view. This data is raw and not processed for other datasets. Kinetics-i3D is a state of the art 3rd person activity recongition model from 2017 trained on a much larger dataset. Kinetics-i3D achieves a much higher classification accuracy than what the ADL model reports. The goal is to use this state of the art model and dataset on ADL to improve the results. Tranfser learning in Kinectics-I3D was achieved with the use of ImageNet. We ask the question, can parameters and mappings learned from the first person point of view be applied to first person?

### Relevant Papers & Git Repos
Kinetics i3D [2017] : https://github.com/deepmind/kinetics-i3d <br />
ADL Paper : https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/
Optical Flow (Using this) : https://github.com/pathak22/pyflow <br />
Other Optical Flow: https://people.csail.mit.edu/celiu/OpticalFlow/ <br />

ADL Indepedent Study PPT: Located within the repo <br />

<b>Related:</b>
Videos: https://arxiv.org/pdf/1406.2199.pdf <br />
Object Detection: https://arxiv.org/pdf/1712.06317.pdf, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kang_Object_Detection_From_CVPR_2016_paper.pdf <br />


### Overview Of Scripts

<b>VideoProcessor.py - </b> Main processing script for the ADL Dataset. Processes video clips for optical flow and RGB input. Instantiate an instance of this class and use the process_clip() method. The method takes as input a Path of type String that represents what directory the processed clip will be stored in as well as a boolean flag denoting whether the video is being processed for training or testing.  < /br>
Example path: inputData/v_doing_laundry_c_9261_time_24899_24963_P_20 <br />
The path format is important, the format is as follows: v_classLabel_c_clip#_time_startFrame_endFrame_VideoName
This just processes one clip at a time in one process. You need a combination of a loop and a bash script that runs multiple processes of this script on different segments of the dataset.<br /><br />
For training data, video clips gets processed as 'x' number of frames of optical flow x, 'x' number of frames of optical flow y, and 'x' number of RGB frames. For Testing Data the video is processed into a numpy array of shape [3][64][224][224][3] for batch size, number of frames, and size of image.<br /><br />

** Note this script inserts code from pyflow into it, and has dependencies with the other scripts inside the VideoProcessing directory. It may only work if all the files are present. Due to time constraints, these dependencies weren't yet resovled. <br /><br />

<i><b>RunVideoProcessor.py - </b></i> The driver script for VideoProcessor.py. Instantiates VideoProcessor and takes in arguments from the command line to process videos. Command line arguments are fed in through a bash script, RunVideoProcessor.sh <br /><br />
<i><b>RunVideoProcessor.sh - </b></i> This bash script runs many instances of VideoProcessor.py to utilize 100% of the CPU. <br /><br />
<i><b>CheckDirectories.py - </b></i> Checks to see what videos have not been processed yet in the particular set you are working with. Compares a text file with all the possible video paths to the actual paths that have been created with processed data inside. The paths in the actual processed video directory that don't exist are printed. <br /><br />
<i><b>MakeDirectories.py - </b></i> Processes the action annotations of the ADL dataset.
For example: 00:30 00:58 14 (mm:ss mm:ss actionLabel, an example annotation from ADL) would create a new path every 64 frames. This path would be just a line in a text file but would later be used to process the actual videol. See the example path above.</p>
<i><b>shuffle.py - </b></i> Shuffles the generated paths from MakeDirectories.Py. Makes 70/20/10 splits for Training/Test/Validation respectively.



##### Modified Kinetics-i3D scripts
<i>Train.py - Modified to load in the correct tensorflow model checkpoints and to save new checkpoints.</i><br />
<i>evaluate_sample.py - </i> Path to input modified<br />
<i>i3d.py - </i> Paths modified, other small edits should be commented. <br />
<i>Config.py - </i> This file specifies the parameters of the network. Modified to work with ADL. See I3D repo for original setup <br /><br />


###### Issues:
- One annotation should not be used in all dataset splits. For example: 00:30 00:58 14 would contain many inputs as the input is only 64 frames right currently. How it is written currently, is that the annotation is processed and then shuffled and split into train/val/test. This causes the data to be too similar in between sets.
- Training crashes when trying to do validation. Seems to be something to do with the Tensorflow Queue. Ignoring validation right now.
- Training appears to work correctly (loss is improving), but testing seems to have some sort of a bug, and predictions are being thrown off.


###### Useful Resources:
- CS231n: Convolutional Neural Networks for Visual Recognition: http://cs231n.stanford.edu/
- Tensorflow CNN tutorial: https://www.tensorflow.org/tutorials/layers
- Andrew Ng Machine Learning tutorials: https://www.youtube.com/watch?v=PPLop4L2eGk


###### Specs:
- Intel Xeon E5-2687W v3 @ 3.10ghz x 20, 128gb 1600mhz Ram, 2TB HDD, Geforce GTX Titan X 12GB
- Python 3.6
- PIL, Numpy, Tensorflow, pyflow. Other library dependencies would be from Kinetics-i3D most likely.

<i>For support, questions, or error reports please contact</i><br />
<i>Ben Hunt: wbh26@txstate.edu</i>
