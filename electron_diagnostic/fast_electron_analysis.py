#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Nov 18 16:51:27 2019

@author: chrisunderwood
    Quick electron analysis
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

import sys
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
import Functions3 as func
from skimage import io
from scipy.signal import medfilt2d


class electron_analysis():
    def __init__(self, filepath, darkfield = None):
        self.filepath = filepath
        self.load_image(darkfield)
        
    def load_image(self, darkfield):
        self.image = io.imread(self.filepath)
        self.image = np.float64(self.image)
        # Remove hard hits
        self.image = medfilt2d(self.image, 5)
        if darkfield is not None:
            self.image -= darkfield
        
    def plot_image(self, vmin = None, vmax= None):
        plt.imshow(self.image, vmin = vmin, vmax = vmax)   
        plt.colorbar()
        plt.show()
        
    def crop_tblr(self, bot, top, left, right):
        self.image = self.image[bot: top, left: right]
        
    def total_charge(self, calibratrion = 1):
        return np.sum(self.image) * calibratrion
        
def sorted_shot_list(folder_path):
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)
    return filelist, shots     

def create_background(shot_numbers):
    if False:
        e = electron_analysis( folderpath + filelist[0])
        bg = np.zeros_like(e.image)
    else:
        bg = np.zeros((2048, 2048))
    
    for bgshot in shot_numbers:
        index = np.where(bgshot == shots)[0][0]
        print (index)
        filepath = folderpath + filelist[index]
        e = electron_analysis(filepath)
        bg += e.image
    bg = bg/(len(shot_numbers))
    return bg
        

if __name__ == "__main__":
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-26/"
    run = "0002/"
    diagnostic = "Lanex/"
    
    folderpath = path_to_data + date + run+ diagnostic
    filelist, shots = sorted_shot_list(folderpath)
    # print (filelist)
    dark_field = create_background([1])
    
    out_dictionary = {}

    for f in filelist[:]:
        print (f)
        shot = f.split("_")[1]
        filepath = folderpath + f
        e = electron_analysis(filepath, dark_field)
        e.crop_tblr(888, 1330, None, 1960)
        # e.plot_image(vmin = 40, vmax = 2500)
        charge = e.total_charge()
        print (charge)
        out_dictionary[shot] = charge
    
    print("Finished Extraction")
    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)
        