# Activities of Daily Living - [First Person Activity Recognition Independent Study Texas State University] 

### Overview
This repository is a compilation of programs to do video processing, particularly on the ADL (Activites of Daily Living) dataset from 2012 to work with the Kinetics Dataset. The ADL dataset consists of 1 million frames of unscripted morning activity in the 1st person point of view. This data is raw and not processed for other datasets. Kinetics-i3D is a state of the art 3rd person activity recongition model from 2017 trained on a much larger dataset. Kinetics-i3D achieves a much higher classification accuracy than what the ADL model reports.

### Relevant Papers & Git Repos
<p>Kinetics i3D [2017] : https://github.com/deepmind/kinetics-i3d</p><br />
<p>ADL Paper : https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/</p><br />

<p>Optical Flow (Using this) : https://github.com/pathak22/pyflow</p><br />
<p>Other Optical Flow: https://people.csail.mit.edu/celiu/OpticalFlow/ </p><br />

<p>Related:</p>
<p> Videos: https://arxiv.org/pdf/1406.2199.pdf </p></br>
<p> Object Detection: https://arxiv.org/pdf/1712.06317.pdf, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kang_Object_Detection_From_CVPR_2016_paper.pdf </p></br>


### Overview Of Scripts

<p><b>VideoProcessor.py - </b> Main processing script for the ADL Dataset. Processes video clips for optical flow and RGB input. Instantiate an instance of this class and use the process_clip() method. The method takes as input a Path of type String that represents what directory the processed clip will be stored in as well as a boolean flag denoting whether the video is being processed for training or testing.  < /br>
Example path: inputData/v_doing_laundry_c_9261_time_24899_24963_P_20<br />
The path format is important, the format is as follows: v_classLabel_c_clip#_time_startFrame_endFrame_VideoName
This just processes one clip at a time in one process. You need a combination of a loop and a bash script that runs multiple processes of this script on different segments of the dataset.</p>
<p><i>RunVideoProcessor.py - </i> The driver script for VideoProcessor.py. Instantiates VideoProcessor and takes in arguments from the command line to process videos. Command line arguments are fed in through a bash script, RunVideoProcessor.sh </p><br />
<p><i>RunVideoProcessor.sh - </i> This bash script runs many instances of VideoProcessor.py to utilize 100% of the CPU. </p><br />
<p><i>CheckDirectories.py</i> Checks to see what videos have not been processed yet in the particular set you are working with. Compares a text file with all the possible video paths to the actual paths that have been created with processed data inside. The paths in the actual processed video directory that don't exist are printed.</p><br />
<p><i>MakeDirectories</i> Processes the action annotations of the ADL dataset.<br />
For example: 00:30 00:58 14 (mm:ss mm:ss actionLabel, an example annotation from ADL) would create a new path every 64 frames. This path would be just a line in a text file but would later be used to process the actual videol. See the example path above.</p><br />

##### Modified Kinetics-i3D scripts
<p><i>Train.py - </i></p><br />
<p><i>evaluate_sample.py - </i></p><br />
<p><i>i3d.py - </i></p><br />


##### Issues:
- One annotation should not be used in all dataset splits. For example: 00:30 00:58 14 would contain many inputs as the input is only 64 frames right currently. How it is written currently, is that the annotation is processed and then shuffled and split into train/val/test. This causes the data to be too similar in between sets.
- Training crashes when trying to do validation. Seems to be something to do with the Tensorflow Queue. Ignoring validation right now.
- Training appears to work correctly (loss is improving), but testing seems to have some sort of a bug, and predictions are being thrown off.



<i>For support, questions, or error reports please contact</i><br />
<i>Ben Hunt: wbh26@txstate.edu</i>
