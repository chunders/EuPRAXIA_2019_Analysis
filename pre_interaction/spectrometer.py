#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Nov 26 23:39:14 2019

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

# Spectrometer analysis.
import sys
sys.path.append("..")
import Functions3 as func
import pandas as pd

class spectrometer():
    def __init__(self, filepath):
        self.filepath = filepath
        self.shot_no = filepath.split("/")[-1].split("_")[1]
        self.df = pd.read_csv(self.filepath, skiprows=3)
        
    def plot_raw_data(self):
        self.df.plot(x='Wavelength')
        
    def crop_spectrum(self, low, high):
        self.df = self.df.loc[(self.df['Wavelength'] >= low) & (self.df['Wavelength'] <= high)]
        
    def plot_spectrum_with_label(self):
        plt.plot(self.df['Wavelength'], self.df['Intensity'], label = self.shot_no)
        
    def return_2col_arr(self):
         return np.c_[spec.df['Wavelength'], spec.df['Intensity']]
        
def sorted_shot_list(folder_path):
    filelist = func.FilesInFolder(folder_path, ".csv")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)
    return filelist, shots     


if __name__ == "__main__":
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-26/"
    run = "0002/"
    diagnostic = "Waves/"
    
    folderpath = path_to_data + date + run+ diagnostic
    filelist, shots = sorted_shot_list(folderpath)    

    out_dictionary = {}

    for f in filelist[:]:
        print (f)
        shot = f.split("_")[1]
        filepath = folderpath + f
        spec = spectrometer(filepath)
        spec.crop_spectrum(775, 825)
        spec.plot_spectrum_with_label()
        out_dictionary[shot] = spec.return_2col_arr()
    plt.legend()
    
    print("Finished Extraction")
    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)    