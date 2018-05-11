from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
import os
import cv2
import collections
import datetime
import time
import argparse
import pyflow
import pathlib


#Base class for processing videos. This video processor is made for processing the ADL dataset
#but can be reused for other purposes.
class VideoProcessor:

	def __init__(self,
					video_path = "ADL_Dataset/P_01.MP4",
					action_list_path = "VideoProcessing/action_annotation/overlap_labels_final.txt",
					action_annotation_path = "VideoProcessing/action_annotation/P_01.txt",
					num_frames = 64,):

		self.video_path = video_path
		self.output_video_path = "" # Is an argument to process_video_clip
		self.action_list_path = action_list_path
		self.action_annotation_path = action_annotation_path
		self.num_frames = 64
		self.action_list = {} # Will populate with the action label
		self.clip_count = 0
		self.number_of_videos = 20
		self.FPS = 30
		self.testData = False



		print(action_list_path)
		#Array index corresponds to label index, string corresponds name of label
		with open(action_list_path) as fin:
			for line in fin.readlines():
				label = line.split("(")[1].split(")")[0] # Get the label
				self.action_list[int(label)] = line.split("'")[1]

		# for debugging
		# for key, value in self.action_list.items():
		# 	print(str(key) + " " + str(value))

		self.annotations = self.get_annotations()
		self.current_label = "" # Will be assigned later




	#Creates Directory with all the videos processed as RGB + Flowx + Flowy at the frame rate specified
	#in the constructor
	#Currently does not loop video, if 64 frames is not available, it will end.
	def process_for_I3D_input(self):

		
		count = 0
		while(count < 20):
			for j in range(len(self.annotations)):
				self.current_label = self.annotations[j].label
				current_start_time = self.annotations[j].start_time
				while(current_start_time <= (self.annotations[j].end_time - (64/30))):
					self.process_video_for_training(current_start_time)
					current_start_time += (64/30)
			count += 1
			self.video_path = self.get_next_video_path()


	#Processes 1 video at the given video path, and saves input to i3D in directory supplied
	#Example path: inputData/v_mopping floor_c_5982_time_17359_17423_P_13
	#clip 5982, start frame: 17359, end frame: 17423, from video P_13d
	def process_clip(self, output_video_path, testData = False):
		

		#To determine when to save processed clips as numpy arrays.
		self.testData = testData

		start_frame = int(output_video_path.split("_")[5])
		end_frame = int(output_video_path.split("_")[6])
		if(testData == False):
			self.output_video_path = output_video_path
			self.process_video_for_training(start_frame)
		else:
			#For test data
			self.output_video_path = "testData/" + output_video_path.split("/")[1]
			print("This is my path " + self.output_video_path)
			self.process_video_for_training(start_frame)
			

	#Returns named tuple of the annotations file
	#start_time - int, end_time - int, label - String
	#time is converted from mm:ss into seconds.
	#label is converted from index to the correpsonding string label name
	def get_annotations(self):

		Annotation = collections.namedtuple("annotation", "start_time end_time label")

		annotations = []

		#Process the annotation file.
		with open(self.action_annotation_path) as fin:
			for line in fin.readlines():
				temp = line.split(" ")

				start_time = 0
				end_time = 0
				label = int(temp[2])

				start_time += (int(temp[0][0])*10 + int(temp[0][1]))*60
				start_time += (int(temp[0][3])*10 + int(temp[0][4]))

				end_time += (int(temp[1][0])*10 + int(temp[1][1]))*60
				end_time += (int(temp[1][3])*10 + int(temp[1][4]))

				#Only add the annotation to the list if it is one of the overlapping labels
				if(label in self.action_list):
					label = self.action_list[label] #Account for index starting at 0
					annotations.append(Annotation(start_time, end_time, label))

		#For debugging
		# for annotation in annotations:
		# 	print(str(annotation.start_time) + " " + str(annotation.end_time) + " " + annotation.label)

		return annotations



	#Start time in frames not seconds
	def process_video_for_training(self,start_frame):

		successFlag = True

		#The shape we want our array to be in
		RGB_Input = np.zeros((1,64,224,224,3))
		FLOW_Input = np.zeros((1,64,224,224,2))

		vidCap = cv2.VideoCapture(self.get_video_path())

		# Get to correct starting frame
		self.skipTillSegment(start_frame,vidCap)

		frameNum = 0

		#Have to read 1 frame in ahead of time to calculate optical flow
		successFlag, frameOld = vidCap.read()
		print(successFlag)
		frameOld = cv2.cvtColor(frameOld, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(frameOld)
		frameOld = self.processImage(frameNum,im)

		while(successFlag and frameNum < 64):
			#Capture frame by frame (frame is a numpy array)
			successFlag, frame = vidCap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			#Shows that is a numpy.ndarray
			print(type(frame))
			print(frame.shape)
			print(successFlag)
			im = Image.fromarray(frame)
			frame = self.processImage(frameNum,im)
			RGB_Input[0][frameNum] = frame


			flowImage = self.calcOpticalFlow(frame,frameOld,frameNum) #flow of image
			FLOW_Input[0][frameNum] = flowImage

			frameOld = frame # for calculating optical flow, you need previous frame
			frameNum += 1


		
		#needs to be an unsigned int for the bytes
		#im = Image.fromarray((RGB_Input[0][149]).astype('uint8'))
		#im.show()

		#Generate the frames and save them from the numpy arrays
		self.makeTrainingInput(RGB_Input,FLOW_Input)

		# print("Before Scaling: " + str(time.time()))
		# #Scale between -1 and 1
		# RGB_Input *= (2.0/255)
		# RGB_Input -= 1
		# print("After: " + str(time.time()))

		# #Save the RGB_Input and Optical Flow input
		# np.save('examples/outFlow.npy', FLOW_Input)
		# np.save('examples/outRGB.npy', RGB_Input)

	#Returns the full path to the next video in the dataset to be processed.
	def get_next_video_path(self,first_half = "ADL_Dataset/P_", extension = ".MP4"):

		vidNumber = int(self.video_path.split("_")[2][:2])
		vidNumber += 1

		if(vidNumber < 10):
			return  first_half + "0" + str(vidNumber) + extension
		
		return  first_half + str(vidNumber) + extension

	def get_next_annotation_path(self,current_path,first_half = "action_annotation/P_", extension = ".txt"):

		vidNumber = int(self.current_path.split("_")[2][:2])
		vidNumber += 1

		if(vidNumber < 10):
			return  first_half + "0" + str(vidNumber) + extension
		
		return  first_half + str(vidNumber) + extension

	def get_video_name(self):

		return self.video_path.split("/")[1].split(".")[0]


	#Gets the current video_path to process from parsing output_video_path.
	#inputData/v_mopping floor_c_5982_time_17359_17423_P_13
	#with the video_path set to "ADL_Dataset/P_01.MP4" would give you "ADL_Dataset/P_13.MP4"
	def get_video_path(self):

		full_path = ""
		temp1 = "".join(self.video_path.split("/")[:-1]) # Ex: ADL_Dataset
		full_path += temp1 + "/"
		temp2 = self.output_video_path.split("_")[-2] + "_" + self.output_video_path.split("_")[-1]# For example P_01
		full_path += temp2 + "."
		temp3 = "".join(self.video_path.split(".")[1]) # Ex: MP4
		full_path += temp3
		print(full_path)

		return full_path




	def get_train_list(self):

		#Starts out getting annotations from P_01.txt
		self.annotations = self.get_annotations()



		with open("inputData/train_list.txt", "w+") as fout:

			count = 0
			while(count < 20):
				for j in range(len(self.annotations)):
					self.current_label = self.annotations[j].label
					current_start_time = self.annotations[j].start_time
					while(current_start_time <= (self.annotations[j].end_time - (64/30))):
						#Make the directory that the 64 flowx,flowy and rgb frames will be saved.
						fout.write("inputData/v_" + self.current_label + "_" + "c_" + str(self.clip_count) + "_" + "time" + "_" + 
							str(int(current_start_time*30)) + "_" + str(int((current_start_time*30)+64)) + "_" +self.get_video_name() + "\n")
						current_start_time += (64/30)
						self.clip_count+=1 #Number of video clip paths made
				count += 1
				if(count != 20):
					self.action_annotation_path = self.get_next_annotation_path(self.action_annotation_path)
					self.annotations = self.get_annotations()







	def process_video_for_evaluation():
		return 0




	#Processes the current image and stores it into the RGB_Input numpy array
	#
	#RGB_Input - the numpy array (input to the neural network 1,79,224,224,3
	#frameNum - the frame number that we are processing, somewhere between 1-79
	#im - the Image object from Pillow library corresponding to the current frame
	def processImage(self,frameNum, im):


		#Make smallest image dimension 256
		if(im.width > im.height):
			im.thumbnail((im.width,256), Image.BILINEAR);
		else:
			im.thumbnail((256,im.height), Image.BILINEAR);

		print(im.size)




		#Crop a 224x224 image out of center and convert to npy array
		horizOffset = (im.width - 224) / 2
		vertOffset = (im.height - 224) / 2

		im = im.crop((horizOffset,vertOffset,
					im.width-horizOffset, im.height-vertOffset))
		#im.show()

		print(im.size)

		imageData = np.array(im)
		#print(imageData.shape)

		#read in the image data (numpy array) into the correct frame index
		return imageData
		#NetworkInput[0][frameNum] = imageData;
		#print(testArray[0][0][0])

		# Some code to help vizualize what is happening with higher
		# dimensionality arrays,TLDR: it's just groups of groups of
		# groups of 2D arrays
		# thinkingDimensions = np.zeros((2,2,2,2,2))
		# print(thinkingDimensions)



	#Calculates the optical flow
	def calcOpticalFlow(self, im1, im2, count):
		im1 = im1.astype(float) / 255.
		im2 = im2.astype(float) / 255.

		# Flow Options:
		alpha = 0.012
		ratio = 0.75
		minWidth = 20
		nOuterFPIterations = 7
		nInnerFPIterations = 1
		nSORIterations = 30
		colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

		s = time.time()
		u, v, im2W = pyflow.coarse2fine_flow(
		    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
		    nSORIterations, colType)
		e = time.time()
		print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
		    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
		flow = np.concatenate((u[..., None], v[..., None]), axis=2)
		return flow
		#np.save('examples/outFlow.npy'+str(count), flow)



	#Skips until segmented part of video (in seconds)
	#startingTime - when in the video the segment starts (pos. integer)
	#vidCap - the inputstream of video from a file.
	#fps - the fps of the video, to calculate how many frames to skip
	def skipTillSegment(self,start_frame,vidCap):

		currentFrame = 0

		#startFrame = fps * startingTime

		successFlag = True

		#Stop 1 frame before startframe so that we have the initial frame to calculate optical flow with
		while(successFlag and currentFrame < start_frame-1):
			#Skip frame by frame
			successFlag, frame = vidCap.read()
			currentFrame += 1



	#Processes 1 clip
	#rgb - [1][64][224][224][3] numpy array
	#flow- [1][64][224][224][2] numpy array
	#This method is to turn these numpy arrays into images. 64 rgb frames, 64 flowx frames, 64 flowy frames
	#Then save these frames into a directory.
	def makeTrainingInput(self,rgb,flow):

		if not os.path.exists(self.output_video_path):
   			os.makedirs(self.output_video_path)
		#Make the directory that the 64 flowx,flowy and rgb frames will be saved.
		#os.mkdir(self.output_video_path)

		#Don't need to process as images if we are doing the test clips, because we need in the form of a numpy array
		if(self.testData == True):
			#Save the RGB_Input and Optical Flow input
			np.save(self.output_video_path + "/" + "out_flow" + ".npy", flow)
			np.save(self.output_video_path + "/" + "out_rgb" + ".npy", rgb)
			return

		#loops through the frames in the form of a numpy array and makes imgs
		for i in range(len(rgb[0])):	
			temp = flow[0][i] # now it is [224][224][2]
			flowx = temp[:,:,0] #Get the flowx portion
			flowy = temp[:,:,1] 

			imFlowx = Image.fromarray(flowx.astype('uint8'),'P')
			imFlowy = Image.fromarray(flowy.astype('uint8'),'P')
			im = Image.fromarray((rgb[0][i]).astype('uint8'))

			imFlowx.save(self.output_video_path + "/" + "flow_x" + str(i) + ".png")
			imFlowy.save(self.output_video_path + "/" + "flow_y" + str(i) + ".png")
			im.save(self.output_video_path + "/" + "img_" + str(i) + ".png")

