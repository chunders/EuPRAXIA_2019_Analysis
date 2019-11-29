import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

logFile = r'Z:\2019 EuPRAXIA\2019-11-29\Untitled.log'
diagList = ['Nearfield pre', 'Farfield pre']

from livePlotting import getLastFileName, getRunFiles, imagesc
import sys
sys.path.append("..\pre_interaction")
from Near_field_analysis import near_field_analysis
from FarField import focal_spot
sys.path.append("..")
import Functions3 as func



import time

from skimage import io
expPath = r'Z:\2019 EuPRAXIA'
oldFilePath = ''

background_file = r"Z:\2019 EuPRAXIA\2019-11-27\0004\Nearfield pre\0004_0003_Nearfield pre.tif"

plt.ion()
f, ax = plt.subplots(nrows=3, gridspec_kw={'height_ratios': [2, 1, 1]} )

tblr = [160, 1150, 290, 1180] # The region on the camera of interest.
NF_dictionary = {}
FF_dictionary = {}

loopCounter = 0

dark_field_NF = io.imread(background_file ) #np.zeros( (2048, 2048) )
umPerPixel = 2.575e-01

while loopCounter < 1e4:
    plt.pause(5)
    filePathList, dateStr, runStr = getLastFileName(logFile,expPath,diagList,returnDateRun=True)
    print("Date and Run ", dateStr+'_'+runStr)
    print (filePathList)
    loopCounter += 1
    if len(filePathList)==2:
        if filePathList[0]==oldFilePath:
            continue
        elif 'fail' not in filePathList[0]:
            filePaths = getRunFiles(logFile,expPath,diagList,dateStr, runStr)
            # print (filePaths)
            print ("filePaths shape", np.shape(filePaths))
            debugStart = None
            for f in filePaths[0][debugStart:]:
                shot = runStr + '__' + str(f.split("\\")[-1].split('_Nearfield')[0])
                print (f, 'key ', shot)

                if not shot in NF_dictionary.keys():
                    try:
                        nf = near_field_analysis(f)
                        nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
                        energy = nf.energy_in_beam(energy_calibration = 2.27e-9)
                        NF_dictionary[shot] = energy 
                    except PermissionError:
                        print ("PermissionError, probs still writing file.")         

            for f in filePaths[1][debugStart:]:
                shot = runStr + '__' + str(f.split("\\")[-1].split('_Nearfield')[0])
                print (f, 'key ',shot)

                if not shot in FF_dictionary.keys():
                    try:
                        ff = focal_spot(f, plot_raw_input=False)
                        fit = ff.fit_2DGaus(umPerPixel, crop_pixels_around_peak = 250, plotting = False)
                        FF_dictionary[shot] = fit 
                    except PermissionError:
                        print ("PermissionError, probs still writing file.")
                    except ValueError:
                        fit = {'amp': [np.nan, np.nan],
                            'xc': [np.nan, np.nan],
                            'yc': [np.nan, np.nan],
                            'sigma_x': [np.nan, np.nan],
                            'sigma_y': [np.nan, np.nan],
                            'theta': [np.nan, np.nan],
                            'offset': [np.nan, np.nan]}
                        FF_dictionary[shot] = fit
            # Put the NF dictionary into lists
            shots = list(NF_dictionary)
            shots_int = []
            energy = []
            # Sort the shots by their run and number
            for s in shots:
                r, sNo = s.split('__')
                shots_int.append(int(r) * 1e3 + int(sNo))  

            shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    
            shots_int_NF = []
            for i, s in enumerate(shots):
                shots_int_NF.append(i)
                energy.append(NF_dictionary[s])

            # Put the FF dictionaries into lists
            shots = list(FF_dictionary)
            shots_int = []
            # Sort the shots by their run and number
            for s in shots:
                s = s.split('_Farfield pre')[0]
                r, sNo = s.split('__')
                shots_int.append(int(r) * 1e3 + int(sNo))  
            shots, shots_int = zip( *sorted( zip(shots, shots_int) ) )                    

            # Create sorted lists
            shots_int_FF = []
            amp = []
            xc = []
            yc = []
            sigma_x = []
            sigma_y = []
            theta = []
            offset = []
            for i, s in enumerate(shots):
                shots_int_FF.append(i)
                amp.append(FF_dictionary[shot]['amp'])
                xc.append(FF_dictionary[shot]['xc'])
                yc.append(FF_dictionary[shot]['yc'])
                sigma_x.append(FF_dictionary[shot]['sigma_x'])
                sigma_y.append(FF_dictionary[shot]['sigma_y'])
                theta.append(FF_dictionary[shot]['theta'])
                offset.append(FF_dictionary[shot]['offset'])
            shots_int_FF = np.array(shots_int_FF)
            amp = np.array(amp )
            xc = np.array( xc )
            yc = np.array( yc )
            sigma_x = np.array( abs(np.array(sigma_x )))
            sigma_y = np.array( abs(np.array(sigma_y )))
            theta = np.array( theta )
            offset = np.array(offset )


            for a in ax:
                a.cla()
            ax[0].plot(shots_int_NF, energy, '.')
            ax[0].set_ylabel("Laser energy (J)")
            ax[-1].set_xlabel("Shot Counter")      
            ax[0].set_ylim([0, 2])  

            func.errorfill(shots_int_FF, xc[:,0], yerr = sigma_x[:,0], ax = ax[1], nstd = 1)
            func.errorfill(shots_int_FF, yc[:,0], yerr = sigma_y[:,0], ax = ax[2], nstd = 1)      
            ax[1].set_ylabel("FF XC\n"+r'$\sigma$: X Width')
            ax[2].set_ylabel("FF YC\n"+r'$\sigma$: Y Width')

            # ax[2].set_ylabel("FF Area")
            # ax[2].plot(shots_int_FF, np.pi * sigma_x[:,0] * sigma_y[:,0])
            plt.suptitle('Pre Interaction ')

            plt.show()

