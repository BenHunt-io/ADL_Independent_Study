#!/bin/bash

for i in {1..2}
	do
		python3 RunVideoProcessor.py $i & 2>&1 >> "processing_log_${i}.txt"
	done

#python3 RunVideoProcessor.py 1 & 2>&1 >> "processing_log_${i}.txt"