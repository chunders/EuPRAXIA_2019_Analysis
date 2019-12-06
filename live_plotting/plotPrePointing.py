import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r'C:\Users\laser\Documents\GitHub')
# C:\Users\laser\Documents\GitHub\EuPRAXIA_2019_Analysis
from EuPRAXIA_2019_Analysis.pre_interaction.FarField import focal_spot

from EuPRAXIA_2019_Analysis.live_plotting.livePlotting import getLastFileName, getRunFiles, imagesc


logFile = r'Z:\2019 EuPRAXIA\2019-12-05\Untitled.log'
diagList = ['Farfield pre']
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''
umPerPixel = 2.575e-01

FF_dictionary = {}

plt.ion()
f, ax = plt.subplots(nrows= 2)

def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[(ctr-N):(ctr)])
    return y/N

while True:
    plt.pause(5)
    
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    #dateStr = '2019-12-04'
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
            for f in filePaths[0][:]:
                shot = runStr + '__' + str(f.split("\\")[-1].split('_Nearfield')[0])
                print (f, 'key ', shot)
                if not shot in FF_dictionary.keys():
                    # Extract data
                    print ('Extracting file' , f)
                    loopCounter = 0
                    while loopCounter < 5:
                        try:
                            fs = focal_spot(f, plot_raw_input = False)
                            loopCounter = 10
                        except PermissionError:
                            plt.pause(1)
                            loopCounter += 1

                        fit = fs.fit_2DGaus(umPerPixel = umPerPixel, crop_pixels_around_peak = 400,
                       plotting = False)
                        FF_dictionary[shot] = fit

            
            # Put the FF dictionary into lists
            shots = list(FF_dictionary)
            print( shots)
            shots_int = []
            energy = []
            # Sort the shots by their run and number
            for i, s in enumerate(shots):
                fileParts =  s.split('__')[1].split('_')
                print (fileParts)
                r = fileParts[0] 
                sNo = fileParts[1]
                shots_int.append(int(r) * 1e3 + int(sNo))  

            shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    
            shots_int_NF = []
            xc = []
            yc = []            
            for i, s in enumerate(shots):
                shots_int_NF.append(i)
                fit = FF_dictionary[s]
                xc.append(fit['xc'])
                yc.append(fit['yc']) 

            xc = np.array(xc)
            yc = np.array(yc)
            nInMean = 5
            xmean = runningMean(xc[:,0], nInMean)
            ymean = runningMean(yc[:,0], nInMean)

            for a in ax:
                a.cla()
            ax[0].plot(np.arange(len(xc[:,0]), dtype = int), xc[:,0], label = 'xc')            
            ax[0].plot(np.arange(len(xc[:,0]))[nInMean:],
                         xmean[nInMean :], '--')

            ax[1].plot(np.arange(len(yc[:,0])), yc[:,0], label = 'yc')  
            ax[1].plot(np.arange(len(yc[:,0]))[nInMean:],
                          ymean[nInMean :], '--')

            plt.suptitle("Pre FF Stability on " +  dateStr)  
            ax[1].set_xlabel('Shots')
            ax[0].set_ylabel('xc')
            ax[1].set_ylabel('yc')
            plt.show()          

                    

                    
