import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

logFile = r'Z:\2019 EuPRAXIA\2019-11-28\Untitled1.log'

diagList = ['Nearfield pre']
from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("../pre_interaction")
from Near_field_analysis import near_field_analysis

import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''

plt.ion()

tblr = [160, 1150, 290, 1180] # The region on the camera of interest.
out_dictionary = {}
while True:
    plt.pause(5)

	filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
	
	print(dateStr+'_'+runStr)
	if len(filePathList)==1:
		if filePathList[0]==oldFilePath:
			continue
		elif 'fail' not in filePathList[0]:
			filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)

			for f in filePathList:
				shot = f.split("_")[1]
				if not 'shot' in out_dictionary.keys():
					nf = near_field_analysis(folder_path  + f, dark_field)
					# nf.plot_image()
					nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])

					energy = nf.energy_in_beam(energy_calibration = 1)
					out_dictionary[shot] = energy
			# Finished extractign data

			shots = list(out_dictionary)
			shots.sort()
			shots_int = []
			energy = []
			for s in shots:
				# if int(s) not in [4, 167]:        
				shots_int.append(int(s))
				energy.append(out_dictionary[s])
			plt.plot(shots_int, energy, '.')
			plt.ylabel("Laser energy (Arb Units)")
			plt.xlabel("Shot Number")
			plt.title(date[:-1] + " run:" + run[:-1])
			plt.ylim([0, None])

			plt.show()
