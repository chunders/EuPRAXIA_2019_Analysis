import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

logFile = r'Z:\2019 EuPRAXIA\2019-11-28\Untitled1.log'
diagList = ['Nearfield pre']

from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("..\pre_interaction")
from Near_field_analysis import near_field_analysis

import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''



plt.ion()

tblr = [160, 1150, 290, 1180] # The region on the camera of interest.
out_dictionary = {}
loopCounter = 0

dark_field = np.zeros( (2048, 2048) )

while loopCounter < 1e4:
    plt.pause(5)
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    print("Date and Run ", dateStr+'_'+runStr)
    loopCounter += 1
    if len(filePathList)==1:
            if filePathList[0]==oldFilePath:
                continue
            elif 'fail' not in filePathList[0]:
                filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
                # print (filePaths, np.shape(filePaths))
                for f in filePaths[0]:
                    shot = runStr + '__' + str(f.split("\\")[-1].split('_Nearfield')[0])
                    print (f, shot)

                    if not shot in out_dictionary.keys():
                        try:
                            nf = near_field_analysis(f)
                            nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
                            energy = nf.energy_in_beam(energy_calibration = 2.27e-9)
                            out_dictionary[shot] = energy 
                        except PermissionError:
                            print ("PermissionError, probs still writing file.")                        

                shots = list(out_dictionary)

                shots_int = []
                energy = []
                # Sort the shots by their run and number
                for s in shots:
                    r, sNo = s.split('__')
                    shots_int.append(int(r) * 1e3 + int(sNo))  

                shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    
                shots_int = []
                for i, s in enumerate(shots):
                    shots_int.append(i)
                    energy.append(out_dictionary[s])

                plt.clf()
                plt.plot(shots_int, energy, '.')
                plt.ylabel("Laser energy (J)")
                plt.xlabel("Shot Number")      
                plt.ylim([0, 2])          
                plt.show()

