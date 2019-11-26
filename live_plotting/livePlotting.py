import numpy as np
import os
import pandas as pd
import glob
import time
import matplotlib
import matplotlib.pyplot as plt

def getLastFileName(logFile,expPath,diagList):
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
        dateStr = df['Date'].values[1]
        runStr = '%04i' % currentRun
        shotStr = '%04i' % currentShot
        for diagStr in diagList:
            matchingFiles = glob.glob(os.path.join(expPath,dateStr,runStr,diagStr,runStr+'_'+shotStr+'_'+diagStr)+'*')
            if len(matchingFiles)==1:
                filePaths.append(matchingFiles[0])
            else:
                filePaths.append('fail')
    return filePaths

        
            
        
