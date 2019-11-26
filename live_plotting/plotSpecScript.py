import matplotlib
import os

import matplotlib.pyplot as plt
logFile = r'Z:\2019 EuPRAXIA\2019-11-26\Untitled1.log'
diagList = ['PostPlasmaSpectrometer']
from livePlotting import continuousPlotting_spectrometer,getLastFileName
from postPlasmaDiagnostics import img2spec
import time
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''
from livePlotting import getLastFileName
from postPlasmaDiagnostics import img2spec

plt.ion()
plt.figure()
ph =[]
while True:
    plt.pause(2)
    filePathList = getLastFileName(logFile,expPath,['PostPlasmaSpectrometer'])
    if len(filePathList)==1:
        if filePathList[0]==oldFilePath:
            continue
        elif 'fail' not in filePathList[0]:

            l,S_l = img2spec(filePathList[0])
            oldFilePath = filePathList[0]
            for h in ph:
                h.remove()
            ph = plt.plot(l,S_l)
            plt.xlabel('Wavelength [nm]')
            plt.title(os.path.split(filePathList[0])[1])
            print(os.path.split(filePathList[0])[1])
            plt.show()
            
           
            
