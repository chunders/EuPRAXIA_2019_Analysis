#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Nov 11 18:37:39 2019

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

# Load my module of functions
import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions3 as func

from skimage import io

class near_field_analysis():
    def __init__(self, filepath, background = None):
        self.filepath = filepath
        self.load_image(background)
        
    def load_image(self, background):
        self.image = io.imread(self.filepath)
        if background is not None:
            self.image -= background
        
    def plot_image(self):
        plt.imshow(self.image)   
        
    def crop_tblr(self, bot, top, left, right):
        self.image = self.image[bot: top, left: right]
        
    def energy_in_beam(self ,energy_calibration):
        energy = np.sum(self.image) * energy_calibration
        # print ("Energy in near field", energy)
        return energy
    
def create_background(shot_numbers):
    if False:
        e = near_field_analysis( folderpath + filelist[0])
        bg = np.zeros_like(e.image)
    else:
        bg = np.zeros((1200,1920))
    
    for bgshot in shot_numbers:
        index = np.where(bgshot == shots)[0][0]
        print (index)
        filepath = folder_path + filelist[index]
        e = near_field_analysis(filepath)
        bg += e.image
    bg = bg/(len(shot_numbers))
    return bg    
    
if __name__ == "__main__":
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-15/"
    run = "0001/"
    diagnostic = "Nearfield post/"
    tblr = [170, 1050, 50, 900]


    # Load all the shot data
    folder_path = path_to_data + date + run + diagnostic
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)
    
    dark_field = create_background([4,167])
    
    
    out_dictionary = {}
    
    for f in filelist[:]:
        shot = f.split("_")[1]
        print (f, shot)
        
        nf = near_field_analysis(folder_path  + f)
        # nf.plot_image()
        # plt.show()
        nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
        # nf.plot_image()
        # plt.show()
        
        energy = nf.energy_in_beam(energy_calibration = 1)
        out_dictionary[shot] = energy
        

    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)        
    
    shots = list(out_dictionary)
    shots.sort()
    shots_int = []
    energy = []
    for s in shots:
        # if int(s) not in [4, 167]:        
        shots_int.append(int(s))
        energy.append(out_dictionary[s])
    plt.plot(shots_int, energy, '.')
    plt.ylabel("Laser energy (Arb Units)")
    plt.xlabel("Shot Number")
    func.saveFigure(path_to_data + date + "Post_laser_energy_NF.png")
