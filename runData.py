#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Feb 10 16:28:10 2020

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

import sys
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
import Functions3 as func


def locateShotFiles(root, day, run, diag, debug = False, fileEnding = ''):
    # print (day, run,  diag)
    files = func.FilesInFolder(root + day + run + diag, fileEnding)
    if debug: print(files)
    try:
        files = sorted(files, key = lambda x : int(x.split("_")[0])*1e4 + int(x.split("_")[1])  )
    except:
        print ("Files cannot be sorted easily")
    if debug: print(files)
    for i, f in enumerate(files):
        files[i] = root + day + run + diag + f
    if debug: print(files)        
    return files
    

def locateRuns(root, day):
    subFs = func.SubFolders_in_Folder(root + day)
    keepers= []
    for f in subFs:
        # print (f)
        try:
            int(f)
            keepers.append(True)
        except:
            # print ("removing: ", f)
            keepers.append(False)
    return np.array(subFs)[keepers]

if __name__ == "__main__":
    rootFolder = '/Volumes/CIDU_passport/2019_Lund_Data/'
    day = '2019-12-03/'
    run = "0002/"
    diaglist = ['Farfield pre', 'Nearfield post', 'PostPlasmaSpectrometer', 
                'Interferometer', 'Nearfield pre', 'Spec_Pre',  'Lanex', 'Phasics']
    diagnostic = 'Nearfield pre'
    
    filePaths = locateShotFiles(rootFolder, day, run,  diagnostic)
    runFolders = locateRuns(rootFolder, day)
