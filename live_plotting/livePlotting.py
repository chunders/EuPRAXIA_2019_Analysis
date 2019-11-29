import numpy as np
import os
import pandas as pd
import glob
import time
import matplotlib
import matplotlib.pyplot as plt


def imagesc(x,y,I,**kwargs):
    ext = (x[0],x[-1],y[0],y[-1])
    ih =plt.imshow(I,extent=ext,aspect='auto',**kwargs)
    return ih

def getLastFileName(logFile,expPath,diagList,returnDateRun=False):
    df = pd.read_csv(logFile,delimiter='\t')
    runNum = df['Run'].values
    shotNum = df['Shot'].values
    currentShot =-1
    currentRun =-1


    if sum(~np.isnan(runNum))>0:
        lastRun = np.nanmax(runNum)
        iSel = (runNum==lastRun)
        lastShot = np.max(shotNum[iSel])
    else:
        lastShot=np.nan
        lastRun=np.nan
    filePaths = []
    if lastShot is not np.nan:
        currentShot = lastShot
        currentRun = lastRun
        dateStr = df['Date'].values[0]
        runStr = '%04i' % currentRun
        shotStr = '%04i' % currentShot
        for diagStr in diagList:
            matchingFiles = glob.glob(os.path.join(expPath,dateStr,runStr,diagStr,runStr+'_'+shotStr+'_'+diagStr)+'*')
            if len(matchingFiles)==1:
                filePaths.append(matchingFiles[0])
            else:
                filePaths.append('fail')
    if returnDateRun:
        return filePaths, dateStr, runStr
    else:
        return filePaths


def getRunFiles(logFile,expPath,diagList,dateStr, runStr):
    df = pd.read_csv(logFile,delimiter='\t')
    runNum = df['Run'].values
    shotNum = df['Shot'].values

    if sum(~np.isnan(runNum))>0:
    
        filePaths = []
        
        for diagStr in diagList:
            matchingFiles = glob.glob(os.path.join(expPath,dateStr,runStr,diagStr,runStr+'_*_'+diagStr)+'*')
            filePaths.append(matchingFiles)
        
    else:
        filePaths=None

    return filePaths      
        

