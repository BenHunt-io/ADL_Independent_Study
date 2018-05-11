import VideoProcessor



def Main():
	print("hello")
	vidProc = VideoProcessor.VideoProcessor()
	#vidProc.process_for_I3D_input()
	#vidProc.get_train_list()
	vidProc.process_clip("inputData/v_mopping floor_c_5982_time_17359_17423_P_13")


if __name__ == "__main__":
	Main()