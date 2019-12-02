import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
logFile = r'Z:\2019 EuPRAXIA\2019-12-02\Untitled.log'
#calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda.mat'
calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda_20191202.mat'
diagList = ['PostPlasmaSpectrometer']
import sys
sys.path.append("..\post_interaction")
print ('Here')
from livePlotting import getLastFileName, getRunFiles, imagesc
from postPlasmaDiagnostics import img2spec, getSpectra
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
    
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,['PostPlasmaSpectrometer'],returnDateRun=True)
    print(dateStr+'_'+runStr)
    if len(filePathList)==1:
        if filePathList[0]==oldFilePath:
            continue
        elif 'fail' not in filePathList[0]:
            filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
            
            l,S_l_list= getSpectra(filePaths[0],calFile)
            S_l = S_l_list[-1]
            #l,S_l = img2spec(filePathList[0])
            oldFilePath = filePathList[0]
            for h in ph:
                ax[0].cla()
            if ih is not None:
                ax[1].cla()
            plt.sca(ax[0])
            ph = plt.plot(l,S_l)
            plt.xlabel('Wavelength [nm]')
            plt.title(os.path.split(filePathList[0])[1])
            print(os.path.split(filePathList[0])[1])
            y = np.arange(len(S_l_list))
            plt.sca(ax[1])
            S_l_series = np.array(S_l_list)
            if np.mean(np.diff(l))<0:
                S_l_series = np.fliplr(S_l_series)
            ih = imagesc(l,y,S_l_series,cmap='inferno')
            
            if k==0:
                plt.colorbar()
                k=1
           
            plt.tight_layout()
            plt.savefig(expPath + '\\' + dateStr + '\\' + runStr + diagList[0] +  'quickAnalysis.png',
                 dpi = 150, bbox_inches='tight')
            plt.show()
            
            
           
