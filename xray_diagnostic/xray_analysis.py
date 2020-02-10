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
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
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
        
    def plot_image(self, vmin = None, vmax = None, show = True):
        plt.imshow(self.image, vmin = vmin, vmax = vmax)   
        plt.colorbar()
        if show:
            plt.show()
        
    def crop_tblr(self, bot, top, left, right):
        self.image = self.image[bot: top, left: right]
        # This might need to move, depending on when the mask is created.
        self.apply_masks()
        
    def apply_masks(self):
        # two mask. background_mask, signal_mask
        
        self.onshot_bg = np.ma.array(self.image, mask= self.signal_mask).mean()
        self.signal = np.ma.array(self.image, mask= self.background_mask)  - self.onshot_bg
        
    def single_hit_spectrum(self, bins = 100, rangeLims = None, plotting = True):
        self.pixel_height = []
        for i in self.signal.data.flatten():
            if abs(i) != 0.0:
                self.pixel_height.append(i)
        self.hist, self.binEdges = np.histogram(self.pixel_height, bins = 100,
                                                range = rangeLims)
        self.bins = 0.5 * (self.binEdges[1:] + self.binEdges[:-1])
        
        
        if plotting:
            plt.plot(self.bins, self.hist)
            # plt.hist(self.pixel_height, bins = bins)     
            plt.show()           
        
        return np.c_[self.bins, self.hist]
        
    
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
        print ()
        raise ValueError('separation of points > diameter \n'+"Min Radius = {}".format(q/2))
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

def create_mask_outline(p1, p2, r, shape):
    mask = np.zeros(shape)    
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
    date = "2019-11-27/"
    run = "0002/"
    diagnostic = "Xray/"
    
    # Load all the shot data
    folder_path = path_to_data + date + run + diagnostic
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)

    # dark_field = io.imread(path_to_data + "2019-11-27/0004/Xray/0004_0003_Xray.tif") 
    
    
    # Cropping to the image.
    crop_path = path_to_data + date + diagnostic[:-1].replace(" ", "_") + "_crop.txt"
    if True:
        # Create crop coors.
        tblr = [300, 1350, 750, 1700]
        topLeft = (441, 1558)
        botRight = (1344, 841)
        wedgeRadius = 670
        # np.savetxt(crop_path, tblr)
    else:
        tblr = np.loadtxt(crop_path, dtype = float)
        tblr = np.array(tblr, dtype = int)    

    
   
    
    if False:
    # =============================================================================
    #     Create background 
    # =============================================================================
        # dark_field = create_background([4,167])
        bg_file_path1 = "/Volumes/Lund_York/2019-11-27/0004/Xray/0004_0003_Xray.tif"
        bg_file_path2 = "/Volumes/Lund_York/2019-11-27/0004/Xray/0004_0047_Xray.tif"    
        bg_files = [bg_file_path1, bg_file_path2]
        dark_field = np.zeros((2048, 2048))
        for file in bg_files:
            bg =  xray_analysis(bg_file_path1)
            bg.image = medfilt2d(bg.image, 11)    
            dark_field += bg.image
        dark_field = dark_field / len(bg_files)
        if False:
            plt.title("Background")
            plt.imshow(dark_field, vmin = np.average(dark_field))
            plt.show()
        np.savetxt(path_to_data + date + "Xray_dark_field.txt", dark_field) 
    else:
        dark_field = np.loadtxt(path_to_data + date + "Xray_dark_field.txt")


    if False:
        # =============================================================================
        #     create mask
        # =============================================================================
        xr = xray_analysis(folder_path  + filelist[-1], 
                            darkfield = dark_field
                           )
        # min_cmap = np.average(xr.image)
        mask = create_mask_outline(topLeft, botRight, wedgeRadius, np.shape(xr.image))
        mask_bg = enlarge_mask_for_background(mask)
    
        xr.plot_image(vmin=None, show = False)
        plt.imshow(mask, alpha = 0.2, cmap = 'Blues')     
        plt.imshow(mask_bg, alpha = 0.15, cmap = 'Reds')
        plt.title("Masks")
        plt.xlim([tblr[2], tblr[3]])
        plt.ylim([tblr[1], tblr[0]])    
        plt.show()
        np.savetxt(path_to_data + date + "Xray_mask.txt", mask,fmt = '%d')
        np.savetxt(path_to_data + date + "Xray_mask_bg.txt", mask_bg, fmt = '%d')   
    else:
        mask = np.loadtxt(path_to_data + date + "Xray_mask.txt", dtype = int)
        mask_bg = np.loadtxt(path_to_data + date + "Xray_mask_bg.txt", dtype = int)    

    out_dictionary = {}
    for f in filelist[:]:
        shot = f.split("_")[1]
        print (f, shot)
        
        xr = xray_analysis(folder_path  + f, 
                           darkfield = dark_field,
                           signal_mask = np.array(mask, dtype = bool),
                           background_mask = np.array(mask_bg, dtype = bool)
                           )
        xr.apply_masks()

        # xr.crop_tblr(tblr[0], tblr[1], tblr[2], tblr[3])

        if False:
            plt.title(shot)
            # xr.plot_image()
            plt.imshow(medfilt2d(xr.signal.data, 11),
                       # cmap = 'Greens'
                       )
            plt.colorbar()
            plt.show()
        
        energy = np.sum(xr.signal)
        out_dictionary[shot] = energy
        
        hist = xr.single_hit_spectrum(rangeLims=(0, 100), bins = 20,
                                      plotting= False)
        out_dictionary[shot] = hist
        
                
        # would like to work out the number of points that are hits.
        # see how close to single hit we are?

    
    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)        
    
    # shots = list(out_dictionary)
    # shots.sort()
    # shots_int = []
    # energy = []
    # for s in shots:
    #     # if int(s) not in [4, 167]:        
    #     shots_int.append(int(s))
    #     energy.append(out_dictionary[s])
    # plt.plot(shots_int, energy, '.')
    # plt.ylabel("Laser energy (Arb Units)")
    # plt.xlabel("Shot Number")
# =============================================================================
#     12 eV makes one count
# =============================================================================
    
    shots = list(out_dictionary)
    hmXrays = []
    shots_int = []
    for s in shots:
        # if int(s) not in [4, 167]:        
        shots_int.append(int(s))
        hmXrays.append(out_dictionary[s][:,1])
    shots_int.append(int(s)+1)      
    hmXrays = np.array(hmXrays)

    # Plot the output.
    for h in hmXrays:
        plt.plot(out_dictionary[s][:,0], h)
    plt.xlim([0, 20])
    plt.xlabel("Energy (Uncal)") 
    plt.yscale('log')
    plt.show()

    plt.pcolormesh(shots_int,out_dictionary[s][:,0], hmXrays.T,
                   norm = mpl.colors.LogNorm())
    plt.xlabel("Shot Number")
    plt.ylabel("Energy (Uncal)")
    plt.colorbar()
    func.saveFigure(path_to_data + date + "Evolution_Xrays.png")
    plt.show()
    
    
   
    
    
    
    