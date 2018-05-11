import os

list_directory = "testData"
training_list  = "training_lists/test_list.txt"

list = os.listdir(list_directory)
print("Number of clips processed: " + str(len(list)))

for path in list:
	print("Type: " + str(type(path)) + path)

size_of_list = len(os.listdir(list_directory))

#See what clips have not been made.
with open(training_list) as fin:
	found = False
	i = 1
	for line in fin.readlines():
		for j in range(len(list)):
			if(str(line.split("/")[1]).strip() == str(list[j]).strip()):
				found = True
				break
		if(not found):
			print("Clip "+ str(i) + ": " + line.split("/")[1])
		i+=1
		found = False

				

