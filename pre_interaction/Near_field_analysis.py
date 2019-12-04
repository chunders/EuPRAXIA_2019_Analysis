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

import sys
sys.path.insert(0, r'C:\Users\laser\Documents\GitHub')


from EuPRAXIA_2019_Analysis import Functions3 as func

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
    
def create_background(shot_numbers, shots):
    # Create a blank image.
    bg = np.zeros((1200,1920))
    
    for bgshot in shot_numbers:
        print (bgshot)
        index = np.where(bgshot == shots)[0][0]
        print (index)
        filepath = folder_path + filelist[index]
        e = near_field_analysis(filepath)
        bg += e.image
    bg = bg/(len(shot_numbers))
    return bg    
    
if __name__ == "__main__":
    dark_field_path = "/Volumes/Lund_York/Dark_fields/"
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-26/"
    run = "0004/"
    diagnostic = "Nearfield pre/"
    crop_path = path_to_data + date + diagnostic[:-1].replace(" ", "_") + "_crop.txt"
    if False:
        # Create crop coors.
        tblr = [160, 1150, 290, 1180]
        # np.savetxt(crop_path, tblr)
    else:
        tblr = np.loadtxt(crop_path, dtype = float)
        tblr = np.array(tblr, dtype = int)

    # Load all the shot data
    folder_path = path_to_data + date + run + diagnostic
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)
    
    # dark_field = create_background([4, 167], shots)
    # dark_field = io.imread(path_to_data + )
    # np.savetxt(dark_field_path + diagnostic[:-1].replace(" ", "_") + ".txt",
    #            dark_field)
    dark_filed = np.loadtxt(dark_field_path + diagnostic[:-1].replace(" ", "_") + ".txt")
    

    
    
    out_dictionary = {}
    
    for f in filelist[:]:
        shot = f.split("_")[1]
        print (f, shot)
        
        nf = near_field_analysis(folder_path  + f, dark_field)
        # nf.plot_image()
        nf.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
        if False:
            nf.plot_image()
            plt.show()
        energy = nf.energy_in_beam(energy_calibration = 1)
        out_dictionary[shot] = energy
        

    print("Finished Extraction")
        
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
    plt.title(date[:-1] + " run:" + run[:-1])
    plt.ylim([0, None])
    func.saveFigure(path_to_data + date + "Evolution_of_laser_energy.png")


    