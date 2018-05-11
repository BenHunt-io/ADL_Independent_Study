import sys
import os
sys.path.append('VideoProcessing')
import VideoProcessor
from config import*
import argparse

def Main():

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('video_clip_num', metavar='N', type=int, help='an integer for the accumulator')

	args = parser.parse_args().video_clip_num

	START_NUM = 14 + (args-1)*8
	END_NUM = 14 + args*8

	if(args > 1):
		START_NUM = 596
		END_NUM = 603
	
	# if(args == 20):
	# 	END_NUM += 6
	

	

	vidProc = VideoProcessor.VideoProcessor()
	videos = []
	input_file_name = TEST_DATA
	#input_file_name train_data.txt
	with open(input_file_name, 'r') as f:
	  for path in f.readlines():
	    path = path.strip()
	    if path:
	      videos.append(path.strip())

	i = 1
	### Process the videos in the given range.
	for video_path in videos:
		# #Skip till the given start video
		if(i < START_NUM):
			i+=1
			continue
		if(i > END_NUM):
			print("break")
			break
		print("Current video: " + str(i) + " " + video_path)
		vidProc.process_clip(video_path, testData = True)
		i+=1

if __name__ == "__main__":
	Main()