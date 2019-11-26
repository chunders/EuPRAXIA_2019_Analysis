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
sys.path.append("..")
import Functions3 as func

class wfs_zernlike():
    def __init__(self, filepath):
        self.read_in_file(filepath)
        
    def read_in_file(self, filepath):
        f = open(filepath, "r")
        lines = f.readlines()
        f.close()
        # for i, l in enumerate(lines):
        #     if i > 1:
        #         print (l.split("\t"))
        names = lines[2][:-1].split("\t")
        data = lines[3][:-1].split("\t")
        print(names, data)
        
        
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

