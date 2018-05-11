import random

#iFileParams = '01'
#iFile = 'testlist' + ifileParams + '.txt'
#oFileTestPath = 'shuffledTestList' + iFileParams + '.txt'
#oFileTrainPath = 'shuffledTrainList' + iFileParams + '.txt'
#oFileValPath = 'shuffledValList' + iFileParams + '.txt'
iFilePath = 'train_list_unshuffled.txt'
oFileTestPath = 'test_list.txt'
oFileTrainPath = 'train_list.txt'
oFileValPath = 'validate_list.txt'

oFileTest = open(oFileTestPath, 'w')
oFileTrain = open(oFileTrainPath, 'w')
oFileVal = open(oFileValPath, 'w')
lines = open(iFilePath).readlines()
count = len(lines)
trainCount = count * .7
testCount = count * .9
random.shuffle(lines)

i = 0
while (i < trainCount):
	oFileTrain.write(lines[i])
	i+= 1
while (i < testCount):
	oFileTest.write(lines[i])
	i+= 1
while (i < count):
	oFileVal.write(lines[i])
	i+= 1


# inputData/v_doing laundry_c_6639_time_45874_45938_P_14
# Extra path that didn't work. 4.17.2018

#inputData/v_reading book_c_8961_time_113827_113891_P_19   From validate data, #560
