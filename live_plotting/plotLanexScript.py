import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

print ('Here')

logFile = r'Z:\2019 EuPRAXIA\2019-11-29\Untitled.log'

diagList = ['Lanex']
from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("..")
# C:\Users\laser\Documents\GitHub\EuPRAXIA_2019_Analysis
from electron_diagnostic.electronDiagnostics import getLaxexSpectra
import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''


plt.ion()
fig,ax = plt.subplots(2,1)
ax = ax.flatten()

ph =[]
ih = None
k=0
while True:
    plt.pause(2)
    
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    # dateStr = '2019-11-28'
    # runStr = '0101'
    # filePathList =  getRunFiles(logFile,expPath,diagList,dateStr,runStr )
    print(dateStr+'_'+runStr)
    if len(filePathList)==1:
        if filePathList[0]==oldFilePath:
            continue
        elif 'fail' not in filePathList[0]:
            filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
            print(filePaths)
            #lineoutList= getLaxexLineouts(filePaths[0])
            eAxis_MeV, specList = getLaxexSpectra(filePaths[0])
            print(eAxis_MeV)
            dE = np.abs(np.mean(np.diff(eAxis_MeV)))
            totalLanex_pC = []
            for l in specList:
                print(np.shape(l))
                totalLanex_pC.append(np.trapz(l,dx=dE))
                
            oldFilePath = filePathList[0]
            print(oldFilePath)
            for h in ph:
                ax[0].cla()
            if ih is not None:
                ax[1].cla()
            plt.sca(ax[0])
            x = np.arange(len(specList))
            # CHANGING AXIS TO BE ENERGY
            # y = np.arange(len(lineoutList[0]))
            y = eAxis_MeV
            lineoutImg = np.flipud(np.swapaxes(np.array(specList),0,1))
            ih = imagesc(x,y,lineoutImg)
            #im = plt.pcolormesh(x,y,lineoutImg)
           
            plt.title("Electrons: " + os.path.split(filePathList[0][-1])[1])
            print(os.path.split(filePathList[0][-1])[1])
            
            plt.sca(ax[1])
            ph = plt.plot(x,totalLanex_pC)
            

            plt.show()
            
            
           
