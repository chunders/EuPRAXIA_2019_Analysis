import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

logFile = r'Z:\2019 EuPRAXIA\2019-12-02\Untitled.log'
diagList = ['Nearfield pre']

from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("..\pre_interaction")
from Near_field_analysis import near_field_analysis
from FarField import focal_spot
sys.path.append("..")
import Functions3 as func

import time

from skimage import io


def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N


expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''

vmin = 1.15
vmax = 1.75
eCal = 2.27e-9 * 0.8157894736842106

background_file = r"Z:\2019 EuPRAXIA\2019-11-27\0004\Nearfield pre\0004_0003_Nearfield pre.tif"

# f, ax = plt.subplots(nrows=1 ) #, gridspec_kw={'height_ratios': [2, 1]} )

tblr = [100, 1200, 400, 1400] # The region on the camera of interest.
NF_dictionary = {}
FF_dictionary = {}

loopCounter = 0

dark_field_NF = io.imread(background_file ) #np.zeros( (2048, 2048) )
runLines = {}

# Load the data from previous runs.
filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
listFilePathsPerRun = []
if int(runStr) > 1:
    maxRun = int(runStr)
    for i in range(1, maxRun):
        mockRun = str(i).zfill(4)
        print ("Run 0 ", mockRun)
        filePaths = getRunFiles(logFile,expPath,diagList,dateStr, mockRun)
        listFilePathsPerRun.append( filePaths[0])
        runLines[mockRun] = len(filePaths[0])



for run in listFilePathsPerRun:
    print (run)
    for fileName in run:
        shot = runStr + '__' + str(fileName.split("\\")[-1].split('_Nearfield')[0])
        print (fileName, 'key ', shot)

        if not shot in NF_dictionary.keys():
            try:
                nf = near_field_analysis(fileName)
                nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
                energy = nf.energy_in_beam(energy_calibration = eCal)
                NF_dictionary[shot] = energy 
            except PermissionError:
                print ("PermissionError, probs still writing file.")     

        print ('Finished ', shot)           
print ('\n\n')
print ("Finished extracting old runs")

shots = list(NF_dictionary)
shots_int = []
energy = []
# Sort the shots by their run and number
for i, s in enumerate(shots):
    r, sNo = s.split('__')
    shots_int.append(int(r) * 1e3 + int(sNo))  

shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    
shots_int_NF = []
for i, s in enumerate(shots):
    shots_int_NF.append(i)
    energy.append(NF_dictionary[s])

plt.ion()
plt.plot(shots_int_NF, energy, '.-')
plt.plot(shots_int_NF, runningMean(energy, 8), '--')


plt.ylabel("Laser energy (J)")
plt.xlabel("Shot Counter")  
plt.ylim([vmin, vmax])  

props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
endOfRun = 0
for r in list(runLines):
    endOfRun += runLines[r]
    plt.vlines(endOfRun, 0, 2.5)
    plt.text(endOfRun, 1.65, r, bbox=props,  rotation = 90)

plt.title('Pre Interaction ' + dateStr)

print (expPath + '\\' + dateStr + '\\' + diagList[0] +  'quickAnalysis.png')
plt.savefig(expPath + '\\' + dateStr + '\\' + diagList[0] +  'quickAnalysis.png',
    dpi = 150, bbox_inches='tight', horizontalalignment='right')
plt.show()
print ("Plotted old runs")



oldRunStr = ''
## Enter While loop
while loopCounter < 1e4:
    plt.pause(5)
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    if oldRunStr is not runStr:
        filePaths = getRunFiles(logFile,expPath,diagList,dateStr, oldRunStr)
        runLines[oldRunStr] = len(filePaths[0])

    print("Date and Run ", dateStr+'_'+runStr)
    print (filePathList)
    loopCounter += 1
    if len(filePathList)==1:
        if filePathList[0]==oldFilePath:
            print ('Old File Path')
            continue
        elif 'fail' not in filePathList[0]:
            try:
                filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
                cont = True
            except pandas.errors.EmptyDataError:
                cont = False
            if cont:
                print ('New File Path')
                print (filePaths)
                print ("filePaths shape", np.shape(filePaths))
                debugStart = None
                for f in filePaths[0][debugStart:]:
                    shot = runStr + '__' + str(f.split("\\")[-1].split('_Nearfield')[0])
                    print (f, 'key ', shot)

                    if not shot in NF_dictionary.keys():
                        try:
                            nf = near_field_analysis(f)
                            nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
                            energy = nf.energy_in_beam(energy_calibration= eCal)
                            NF_dictionary[shot] = energy 
                        except PermissionError:
                            print ("PermissionError, probs still writing file.")         

                # Put the NF dictionary into lists
                shots = list(NF_dictionary)
                shots_int = []
                energy = []
                # Sort the shots by their run and number
                for i, s in enumerate(shots):
                    r, sNo = s.split('__')
                    shots_int.append(int(r) * 1e3 + int(sNo))  


                shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    
                shots_int_NF = []
                for i, s in enumerate(shots):
                    shots_int_NF.append(i)
                    energy.append(NF_dictionary[s])



                plt.clf()
                plt.plot(shots_int_NF, energy, '.-')
                plt.plot(shots_int_NF, runningMean(energy, 8), '--')

                plt.ylabel("Laser energy (J)")
                plt.xlabel("Shot Counter")  


                plt.ylim([vmin, vmax])  

                endOfRun = 0
                for r in list(runLines):
                    if r is not oldRunStr:
                        endOfRun += runLines[r]
                        plt.vlines(endOfRun, 0, 2.5)
                        plt.text(endOfRun, vmin, r, bbox=props, rotation = 90)

                plt.title('Pre Interaction ' + dateStr)
                plt.savefig(expPath + '\\' + dateStr + '\\' + diagList[0] +  'quickAnalysis.png',
                   dpi = 150, bbox_inches='tight', horizontalalignment='right')
                plt.show()
                oldRunStr = runStr + ''
