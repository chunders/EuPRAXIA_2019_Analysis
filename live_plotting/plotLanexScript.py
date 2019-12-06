import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap


if False:
    import datetime

    now = str(datetime.datetime.now())
    TodaysDate = now.split(" ")[0]
    # logFile = r'Z:\2019 ' + 'EuPRAXIA\{}\Untitled.log'.format(TodaysDate)
else:
    logFile = r'Z:\2019 EuPRAXIA\2019-12-05\Untitled.log'



diagList = ['Lanex']
from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("..")
# C:\Users\laser\Documents\GitHub\EuPRAXIA_2019_Analysis
from electron_diagnostic.electronDiagnostics import getLaxexSpectra
import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''

def LEGENDCmap():
    colors = [(1,1,1), (.6,.6,.85),(.1,.1,.7),(.2,.7,1),(.2,1,.2),(1,1,.2),(1,.5,0),(1,0,0),(.7,0,0),(.2,0,0)]          
    nbins = 256
    cmapName = "LEGEND"
    cmap = LinearSegmentedColormap.from_list(cmapName,colors,nbins)
    return cmap


import argparse
parser = argparse.ArgumentParser(description='Lower and Upper Charge')
parser.add_argument('-l', '--lower', default=1e-3, type=float)
parser.add_argument('-u', '--upper', default=2, type=float)
args = parser.parse_args()

upperEnergy = 275



maxCharge_pC = args.upper
minCharge_pC = args.lower

plt.ion()
fig,ax = plt.subplots(2,1)
ax = ax.flatten()


ph =[]
ih = None
k=0
file_dictionary = {}
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
            loopCounter = 0
            while loopCounter < 5:
                try:
                    eAxis_MeV, specList, file_dictionary = getLaxexSpectra(filePaths[0], file_dictionary)
                    loopCounter = 10
                except PermissionError:
                    plt.pause(1)
                    loopCounter += 1
                    

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
            ih = imagesc(x,y,lineoutImg, vmin = minCharge_pC, vmax = maxCharge_pC,
                    #  norm = matplotlib.colors.LogNorm(),
                    cmap = LEGENDCmap()
                     )
            if k==0:
                cbar = plt.colorbar()
                cbar.set_label('Charge (pC)')
                k=1
            plt.ylabel('Energy (MeV)')
            plt.xlabel('Shot ')
            plt.ylim([y[0], upperEnergy])
            #im = plt.pcolormesh(x,y,lineoutImg)
           
            plt.title("Electrons: " + runStr)
            print(os.path.split(filePathList[0][-1])[1])
            
            plt.sca(ax[1])
            ph = plt.plot(x,totalLanex_pC)
            plt.ylabel("Charge (pC)")
            plt.xlabel('Shot ')
            plt.tight_layout()
            plt.savefig(expPath + '\\' + dateStr + '\\' + runStr + diagList[0] +  'quickAnalysis.png',
                 dpi = 150, bbox_inches='tight')            
            plt.show()
            
            
