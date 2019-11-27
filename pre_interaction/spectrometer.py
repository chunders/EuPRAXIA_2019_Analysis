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
from  select_best_gaussFit_class import select_Gaus_fit
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d

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
         return np.c_[self.df['Wavelength'], self.df['Intensity']]
     
    def fourier_pulse_duration(self, plotting = False):
        lambda_spec = self.df['Wavelength']
        nos_data_points = len(lambda_spec)
        y = self.df['Intensity']
        c = 3e8
        freq = c / lambda_spec
        freq_even = np.linspace(freq.min(), freq.max(), nos_data_points * 10)
        int_func = interp1d(freq, y, kind = 'cubic')
        new_freq_intensity = int_func(freq_even)
        # plt.plot(freq, y)
        # plt.plot(freq_even, new_freq_intensity)        
        # plt.show()
        Fy = np.fft.fft(new_freq_intensity)
        n = len(new_freq_intensity)//2
        Fy = np.concatenate( (Fy[n:],Fy[:n]), axis = 0)
        
        view_size = 250
        x = np.arange(n - view_size, n + view_size)
        f_crop = abs(Fy)[n - view_size: n + view_size]
        # plt.plot(x, f_crop)
        # plotting = False
        g = select_Gaus_fit(x, f_crop, 
                    plottingOnOff=plotting, guess=[f_crop.max(), n, 40., 0.] )
        if plotting:
            plt.show()
        fwhm = g.fwhm_of_best()
        
        if plotting:
            f, ax = plt.subplots(nrows = 2)
            ax[0].plot(abs(Fy), 'r.-')
            # ax[0].set_xlim([800, 1700])
            # ax[0].set_xlim([1220, 1300])
            g.BestFit(ax = ax[0], color = 'b', lw = 3)
            ax[0].set_xlim([n - view_size, n + view_size]) 
            func.scientificAxis('y', ax[0])
            ax[0].set_ylabel("Fourier Space Amp")   
            
            # 
            ax[1].plot(lambda_spec, y, 'b')
            ax[1].set_xlabel("Wavelenth (nm)")
            ax[1].set_ylabel("Intensity")   
            ax[1].set_xlim([775, 825])            
            
            plt.show()
            
            
        return fwhm
            

        
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
    fit_dictionary = {}
    

    for f in filelist[:]:
        print (f)
        shot = int(f.split("_")[1])
        filepath = folderpath + f
        spec = spectrometer(filepath)
        # spec.crop_spectrum(775, 825)
        cropped_spec = spec.return_2col_arr()
        
        out_dictionary[shot] = cropped_spec
        g = select_Gaus_fit(cropped_spec[:,0],cropped_spec[:,1], 
                            plottingOnOff=False, guess=[400., 800., 40., 10.] )
        Ffwhm = spec.fourier_pulse_duration()
        fit_dictionary[shot] = [g.fwhm_of_best(),Ffwhm]
        # plt.show()
        print()
    
    ## If plotting each spectrum uncomment.
    #     spec.plot_spectrum_with_label()
    # plt.legend()
    # plt.show()
    

    print("Finished Extraction")
    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)    

    
    hm = []
    data_shots = list(out_dictionary)
    data_shots.sort()
    for shot in data_shots:
        hm.append(medfilt(out_dictionary[shot][:,1], 7))
    wavelength = out_dictionary[shot][:,0]
    plt.pcolormesh(wavelength, data_shots,
                   hm)
    plt.colorbar()
    plt.show()
    
    # g = select_Gaus_fit(wavelength, 1000 + np.array(hm[-1]), plottingOnOff=True,
    #                     guess=[400., 800., 40., 10.] )
    # g_result = g.BestFit()
    # fwhm = g.fwhm_of_best()
    # print (g_result[1:])
    
    # Plot fitted dictionary
    lists = sorted(fit_dictionary.items()) # sorted by key, return a list of tuples
    shots, fwhm_results = zip(*lists) # unpack a list of pairs into two tuples
    fwhm_results = np.array(fwhm_results)
    f, ax = plt.subplots(nrows = 2)
    ax[0].plot(shots, fwhm_results[:,0], '.-')
    ax[1].plot(shots, fwhm_results[:,1], '.-') 
    ax[0].set_ylim([0, None])
    ax[1].set_ylim([0, None])    
    plt.suptitle(date[:-1] + " run:" + run[:-1] + " Spectrum Pre Interaction")
    ax[0].set_ylabel("FWHM (nm)")
    ax[1].set_ylabel(r"Fourier limit $\tau$ (arb units)")    
    ax[-1].set_xlabel("Shot")
    plt.show()
    
