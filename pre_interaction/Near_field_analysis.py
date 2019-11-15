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
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_image()
        
    def load_image(self):
        self.image = io.imread(self.filepath)
        
    def plot_image(self):
        plt.imshow(self.image)   
        
    def crop_tblr(self, bot, top, left, right):
        self.image = self.image[bot: top, left: right]
        
    def energy_in_beam(self ,energy_calibration):
        energy = np.sum(self.image) * energy_calibration
        print ("Energy in near field", energy)
        return energy
    
    
    
if __name__ == "__main__":
  folderPath = "/Volumes/GoogleDrive/My Drive/2019_Lund/Pre_plasma_diagnositc_calibration/"
  nf = near_field_analysis(folderPath  + "Near_field.tif")
  nf.crop_tblr(400, 1100, 700, 1600)
  nf.plot_image()

nf.energy_in_beam(energy_calibration = 1)
