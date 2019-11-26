#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Nov 20 20:32:43 2019

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import Functions3 as func

from scipy.signal import medfilt2d, convolve2d
from skimage import io, draw
from collections import namedtuple

Pt = namedtuple('Pt', 'x, y')
Circle = Cir = namedtuple('Circle', 'x, y, r')

class xray_analysis():
    def __init__(self, filepath, median_filter_kernel = 11, 
                 darkfield = None, 
                 signal_mask = None,
                 background_mask = None, bg_med_filter = 11):
        self.filepath = filepath
        self.load_image(darkfield)
        self.median_filter_kernel = median_filter_kernel
        print (median_filter_kernel)
        self.image = medfilt2d(self.image, self.median_filter_kernel)
        
        self.signal_mask = 1
        self.background_mask = 0
        # self.bg_med_filter = bg_med_filter
        
        if signal_mask is not None:
            self.signal_mask = signal_mask
            
        if background_mask is not None:
            self.background_mask = background_mask   
                    
            
        
    def load_image(self, background):
        self.image = io.imread(self.filepath)
        if len(np.shape(self.image)) > 2:
             self.image = self.image[:,:,0]
        self.image = np.float64(self.image)
             
        if background is not None:
            self.image -= background
        
    def plot_image(self):
        plt.imshow(self.image)   
        plt.colorbar()
        plt.show()
        
    def crop_tblr(self, bot, top, left, right):
        self.image = self.image[bot: top, left: right]
        # This might need to move, depending on when the mask is created.
        self.apply_masks()
        
    def apply_masks(self):
        self.onshot_bg = np.average(self.background_mask * self.image)
        self.signal = self.signal_mask * (self.image - self.onshot_bg)        
        
    
def create_background(shot_numbers):
    if False:
        x = xray_analysis(folderpath + filelist[0])
        bg = np.zeros_like(x.image)
    else:
        bg = np.zeros((1200,1920))
    
    for bgshot in shot_numbers:
        index = np.where(bgshot == shots)[0][0]
        print (index)
        filepath = folder_path + filelist[index]
        x = xray_analysis(filepath)
        bg += x.image
    bg = bg/(len(shot_numbers))
    return bg    

    

 


def circles_from_p1p2r(p1, p2, r):
    'Following explanation at http://mathforum.org/library/drmath/view/53027.html'
    if r == 0.0:
        raise ValueError('radius of zero')
    (x1, y1), (x2, y2) = p1, p2
    if p1 == p2:
        raise ValueError('coincident points gives infinite number of Circles')
    # delta x, delta y between points
    dx, dy = x2 - x1, y2 - y1
    # dist between points
    q = (dx**2 + dy**2)**0.5
    if q > 2.0*r:
        raise ValueError('separation of points > diameter')
    # halfway point
    x3, y3 = (x1+x2)/2, (y1+y2)/2
    # distance along the mirror line
    d = (r**2-(q/2)**2)**0.5
    # One answer
    c1 = Cir(x = x3 - d*dy/q,
             y = y3 + d*dx/q,
             r = abs(r))
    # The other answer
    c2 = Cir(x = x3 + d*dy/q,
             y = y3 - d*dx/q,
             r = abs(r))
    return c1, c2    

def create_mask_outline(p1, p2, r):
    mask = np.zeros((tblr[1]-tblr[0], tblr[3]-tblr[2]))    
    c1, c2 = circles_from_p1p2r(p1, p2, r)  
    for i, c in enumerate( [c1, c2] ):
        rr, cc = draw.circle(c.x, c.y, radius= float(c.r) + i * -10,
                                   shape=mask.shape)   
        mask[rr, cc] += 1   
    mask_bool = mask > 1
    mask_ones = np.array(mask_bool, dtype = float)        
    return mask_ones
    

def enlarge_mask_for_background(mask_ones, size = (20,20)):

    square = np.ones(size)
    outside_mask = convolve2d(mask_ones, square) > 0
    outside_mask = outside_mask[square.shape[0]//2:-square.shape[0]//2 +1,
                                square.shape[1]//2:-square.shape[1]//2 +1]
    outside_mask_inv = np.array(outside_mask==False, dtype = float)
    return outside_mask_inv

if __name__ == "__main__":
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-15/"
    run = "0001/"
    diagnostic = "Xray/"
    
    # Cropping to the image.
    tblr = [680, 1200, 500, 1000]

    
    # Load all the shot data
    folder_path = path_to_data + date + run + diagnostic
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)    
    
    
    
    bg =  xray_analysis(folder_path + "0001_0296_Xray.tif")
    bg.crop_tblr(1000, None, 1000, None)
    bg.image = medfilt2d(bg.image, 11)
    plt.title("Background")
    bg.plot_image()
 
    background = np.average(bg.image)


    
    # dark_field = create_background([4,167])
    
# =============================================================================
#     create mask
# =============================================================================
    mask = create_mask_outline((453, 136), (50, 412), 295)
    mask_bg = enlarge_mask_for_background(mask)
    
    plt.imshow(mask, alpha = 0.4, cmap = 'Blues') 
    plt.imshow(mask_bg, alpha = 0.3, cmap = 'Reds')
    plt.title("Masks")
    plt.show()
    

    xr = xray_analysis(folder_path  + f, darkfield = background)
    xr.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])
    # xr.image = medfilt2d(xr.image, 5)


    # plt.imshow(xr.image, cmap = 'nipy_spectral')    
    # plt.colorbar()    
    # # plt.imshow(mask_bool, cmap = 'Reds', alpha = 0.2)
    # plt.imshow(mask, alpha = 0.4, cmap = 'Blues') 
    # plt.imshow(mask_bg, alpha = 0.3, cmap = 'Reds')    
    # plt.show()
    

    out_dictionary = {}
    
    for f in filelist[:3]:
        shot = f.split("_")[1]
        print (f, shot)
        
        xr = xray_analysis(folder_path  + f, 
                           darkfield = background,
                           signal_mask = mask,
                           background_mask = mask_bg)

        xr.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])

        if True:
            plt.title(shot)
            # xr.plot_image()
            plt.imshow(medfilt2d(xr.signal, 11),
                       cmap = 'Greens')
            plt.colorbar()
            plt.show()
        
        energy = np.sum(xr.signal)
        out_dictionary[shot] = energy
        
        
        # would like to work out the number of points that are hits.
        # see how close to single hit we are?

    
'''        
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
    func.saveFigure(path_to_data + date + "Evolution_of_laser_energy.png")
'''
    