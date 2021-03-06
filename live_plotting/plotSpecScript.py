import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt


if True :
    import datetime
    now = str(datetime.datetime.now())
    TodaysDate = now.split(" ")[0]
    logFile = r'Z:\2019 ' + 'EuPRAXIA\\{}\\Untitled.log'.format(TodaysDate)
    calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda_' + '{}.mat'.format( TodaysDate.replace('-', ''))

else:
    logFile = r'Z:\2019 EuPRAXIA\2019-12-05\Untitled.log'
    calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda_20191205.mat'


#calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda.mat'
diagList = ['PostPlasmaSpectrometer']
import sys
sys.path.append("..\post_interaction")
from livePlotting import getLastFileName, getRunFiles, imagesc
from postPlasmaDiagnostics import img2spec, getSpectra
import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''

print ("Starting on day ",TodaysDate )
plt.ion()
fig, ax = plt.subplots(nrows = 2)

ExtractPrevious = True

ph =[]
ih = None
k=0
while True:
    plt.pause(2)
    
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    print(dateStr+'_'+runStr, filePathList)

    if ExtractPrevious:
        print ('Extracting run data')
        filePathList = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
        print (dateStr, runStr, filePathList)
        ExtractPrevious = False

    if len(filePathList)==1:
        if filePathList[0]==oldFilePath:
            continue
        elif 'fail' not in filePathList[0]:

            print ('Extracting Data')
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

            print ('Plotting the following arrays')
            print (l,S_l)

            plt.xlabel('Wavelength [nm]')
            print (filePathList )
            # print(os.path.split(filePathList[0])[1])
            # plt.title(os.path.split(filePathList[0])[1])
            plt.title(runStr)
            

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
            
            
           
