#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Nov 18 17:17:36 2019

@author: chrisunderwood
    Analysis of the wavefront sensor
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

import sys
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
import Functions3 as func

class wfs_zernlike():
    def __init__(self, filepath):
        self.data = self.read_in_file(filepath)
        
    def read_in_file(self, filepath, printing = False):
        with open(filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        names = lines[2][:-1].split("\t")
        data = lines[3][:-1].split("\t")
        data = [dat.replace(",", ".") for dat in data]
        
        data = np.array(data, dtype = float)
        outDictionary = {}
        for n, d in zip(names, data):
            if printing: print ("{}\t{}".format(n, d))
            outDictionary[n] = d
            
        return outDictionary
        
if __name__ == "__main__":    
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-15/"
    run = "0001/"
    diagnostic = "Phasics/"
    folder_path = path_to_data + date + run + diagnostic
    
    filelist = func.FilesInFolder(folder_path, ".txt")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)  
    
    for f in filelist[:1]:
        print (f)    
        wfs = wfs_zernlike(folder_path + f)
        print (wfs.data)

