#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Mar 21 16:29:33 2018

@author: chrisunderwood

    A file for doing the analysis on a far field
    
    
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

import sys
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
import Functions3 as func


imageRead = True
if imageRead:
    from skimage import io
else:
    from PIL import Image


class focal_spot(): 
    #Takes the folder path and then the file name
    def __init__(self, filepath, calculations = True,
                 backgroundImage = None, plot_raw_input = True):
        # The image should be a numpy array
        self.filepath = filepath
        self.load_image(backgroundImage)
                    
        if calculations:
            self.create_class_variables()
            
        if plot_raw_input:
            plt.imshow(self.im)
            plt.colorbar()
            plt.show()
    
    def load_image(self, backgroundImage):
        self.im = io.imread(self.filepath)    
        if backgroundImage is not None:
            # Check that the image is a float before taking the background away
            self.im = np.float64(self.im) -  np.float64(np.array(backgroundImage))

                        
    def create_class_variables(self):
        self.imShape = np.shape(self.im)
        self.background_subtraction()
        self.normalise()
#        self.maxCoors = np.unravel_index(self.im.argmax(), self.im.shape)
        self.Peaklocator()
        self.ThresholdImage()
        self.lineOutIntegrals()
        
    def background_subtraction(self):
        # Basic background subtraction, zero the lowest value
        self.im = self.im - self.im.min()

    def normalise(self):
        FS_im_bg = (self.im - self.im.min())
        self.im_bg_n = (FS_im_bg * 1.0) / FS_im_bg.max()        
        
    def Peaklocator(self, plotting = False):
        self.ylineout = []
        for im in self.im:
            self.ylineout.append(sum(im))
        self.xlineout = []
        for im in self.im.T:
            self.xlineout.append(sum(im))
#        print self.maxCoors
        self.maxCoors = [self.xlineout.index(max(self.xlineout)), 
                         self.ylineout.index(max(self.ylineout))]
        print("Maximum Coordinate (pixel nos): ", self.maxCoors)
        if plotting:
            plt.plot(range(len(self.ylineout)), self.ylineout)
            plt.plot(range(len(self.xlineout)), self.xlineout)
            plt.show()
            
    def ThresholdImage(self):
        # Create an image thresholded to 50%
        ThresLimit = 0.5
        mask = self.im_bg_n > ThresLimit
        self.thresIm = mask * self.im_bg_n

    def lineOutIntegrals(self):
        # Create lineouts of the thresholded image, so 
        self.sumY = []
        for im in self.thresIm:
            self.sumY.append(sum(im))
        self.sumX = []
        for im in self.thresIm.T:
            self.sumX.append(sum(im))
    
    def lineFWHM(self, line):
#==============================================================================
# THIS FUNCTION NEEDS CHECKING
# New method with np.nonzero seems to work well.
#==============================================================================
        # Method 1: see the first place when line is not none. Should measure 
        # weirded shaped spots better, and catch outlining regions of high intensity
        start = next((i for i, x in enumerate(line) if x), None)
        fin = len(line) - next((i for i, x in enumerate(line[::-1]) if x), None) 

        # Method 2: quick and cannot fail
        d = np.nonzero(line)
        diff = np.shape(d)[1] - (fin-start) 
        if diff > 1:
            print ("Method 1: " , np.shape(d)[1])
            print ("Method 2: " , fin-start)
            print ("Difference: {}\n".format(diff))
            print (self.fp)
        return np.shape(d)[1] #fin-start    
    
    def calcVals(self, umPerPixel):
        if True:
            plt.plot(self.sumX, label = "x")
            plt.plot(self.sumY, label = "y")
            plt.legend()
            plt.show()
        self.fwhmX = self.lineFWHM(self.sumX)
        self.fwhmY = self.lineFWHM(self.sumY)
        self.fwhmRatio = self.thresIm.sum() / self.im_bg_n.sum()  
        return str(self.fwhmX*umPerPixel) +'um ' +str(self.fwhmY*umPerPixel) +'um ' + str((self.fwhmX*umPerPixel + self.fwhmY*umPerPixel )/2.0) + 'um ' +str( self.fwhmRatio) + '\n'
    
    # the calibration of pixels to length (microns)
    def valArray(self, umPerPixel):
        return [self.fwhmX*umPerPixel, self.fwhmY*umPerPixel, (self.fwhmX*umPerPixel + self.fwhmY*umPerPixel )/2.0, self.fwhmRatio]
    
    
    def createPlotWithLineOuts(self, imageSize = 120,  title = ''):
       
        fig = plt.figure(figsize=(8,6))
        # 3 Plots with one major one and two extra ones for each axes.
        gs = gridspec.GridSpec(4, 5, height_ratios=(1,1,1,1), width_ratios=(3,3,3,3,0.5))
        gs.update(wspace=0.025, hspace=0.025)
        ax1 = plt.subplot(gs[0:3, 0:3])             # Image
        ax2 = plt.subplot(gs[0:3, -2] , sharey=ax1) # right hand side plot
        ax3 = plt.subplot(gs[-1, 0:3] , sharex=ax1) # below plot
#        plt.title('class')
        ax1.set_xlim(self.maxCoors[0]-imageSize, self.maxCoors[0]+imageSize)
        ax1.set_ylim(self.maxCoors[1]-imageSize, self.maxCoors[1]+imageSize)

        
#        print np.shape(self.im_bg_n)
        CS = ax1.contour(self.im_bg_n, 
            levels = [1/np.e**2, 1/np.e, 0.5, 0.9])
        text = ['1/e**2', '1/e', '0.5', '0.9']
        ax1.clabel(CS, fmt='%.1f', colors='k', fontsize=14)        
        
        for i in range(len(text)):
            CS.collections[i].set_label(text[i])
        
        ax1.legend(loc='upper left')
        
        ax1.imshow(self.im, cmap='Blues')
#        ax1.vlines(self.maxCoors[1], 0, self.imShape[0])
#        ax1.hlines(self.maxCoors[0], 0, self.imShape[1])
        ax2.plot(self.sumY, list(range(len(self.sumY))))
        ax3.plot(list(range(len(self.sumX))), self.sumX )
        plt.suptitle(title, y=0.95, fontsize = 18)
        ax1.set_ylabel('Pixels')
        ax3.set_xlabel('Pixels')
        
        plt.show()
        
    def return_energy(self):
        return np.sum(self.im)
        
    
    def plot_scaledImage(self, umPerPixel):
        plt.pcolormesh(np.arange(self.imShape[1]) * umPerPixel, 
                       np.arange(self.imShape[0]) * umPerPixel, 
                      self.im, cmap='Blues')
        plt.colorbar()
        plt.xlabel("Distance $(\mu m)$")
        plt.ylabel("Distance $(\mu m)$")
        plt.show()
        
    def twoD_Gaussian(self,xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        (x, y) = xdata_tuple                                                        
        xo = float(xo)                                                              
        yo = float(yo)                                                              
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                            + c*((y-yo)**2)))                                   
        return g.ravel()
    

    def fit_2DGaus(self,umPerPixel, plotting = True, view_n_std = 4, 
                   crop_pixels_around_peak = False):
        ''' crop_pixels_around_peak should be replaced with the number of pixels to crop by
        '''
        
        if crop_pixels_around_peak == False:
            shape = self.imShape
            image = self.im
            maxCoors = self.maxCoors
            guessWhole = True
        else:
            image = self.im[self.maxCoors[1] - crop_pixels_around_peak: self.maxCoors[1] + crop_pixels_around_peak,
                            self.maxCoors[0] - crop_pixels_around_peak: self.maxCoors[0] + crop_pixels_around_peak]
            shape = np.shape(image)
            maxCoors = np.array([crop_pixels_around_peak //2, crop_pixels_around_peak//2])
            guessWhole = False
            
        # print ('In fit 2d gaus')
        # print ('shape', shape, 'max coors', maxCoors)


        self.umPerPixel = umPerPixel
        
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        x, y = np.meshgrid(x, y)
        if guessWhole:
            #               amplitude, xo, yo 
            initial_guess = [image.max(), maxCoors[1], maxCoors[0],
                         # sigma_x, sigma_y, theta, offset
                         5, 5, 0, 0]
        else:
            initial_guess = [image.max(), crop_pixels_around_peak, crop_pixels_around_peak,
                         # sigma_x, sigma_y, theta, offset
                         5, 5, 0, 0]            
        popt, pcov = curve_fit(self.twoD_Gaussian, (x, y), image.T.ravel(), p0=initial_guess)
        perr = func.pcov_to_perr(pcov)
        
        data_fitted = self.twoD_Gaussian((x, y), *popt)

        if plotting:
            plt.pcolormesh(x, y, image.T, cmap = "Reds")
            plt.contour(x, y, data_fitted.reshape(shape[1], shape[0]),
                        5, #colors='w',
                        cmap = 'jet'
                        )
            xLims = [popt[1] - view_n_std * popt[3], popt[1] + view_n_std * popt[3]]
            yLims = [popt[2] - view_n_std * popt[4], popt[2] + view_n_std * popt[4]]            
                
            for lim, shape  in zip([xLims, yLims], shape):
                if lim[0] < 0:
                    lim[0] = 0
                if lim[1] > shape -1:
                    lim[1] = shape - 1
                    
            plt.xlim(xLims)
            plt.ylim(yLims)
            plt.show()

        print ("Fit Params:")
        names = ["amplitude", "xo [um]   ", "yo [um]   ", "sigma_x [um]", "sigma_y [um]", "theta    ", "offset    "]
        dnames = ["amp", "xc", "yc", "sigma_x", "sigma_y", "theta", "offset"]        
        TF = [False, True, True, True, True, False, False]
        out_dict= {}
        for n, nd, out, err, tf in zip(names, dnames, popt, perr, TF):
            if tf:
                out, err = np.array([out, err]) * self.umPerPixel
            print ("{}\t\t{:2.2f} +/- {:1.2f}".format(n, out, err))
            out_dict[nd] = [out, err]
        
        print ("Fit Params:")
        names = ["amplitude", "xo [um]   ", "yo [um]   ", "sigma_x [um]", "sigma_y [um]", "theta    ", "offset    "]
        dnames = ["amp", "xc", "yc", "sigma_x", "sigma_y", "theta", "offset"]        
        TF1 = [False, True, True, True, True, False, False]
        TF2 = [False, 'x', 'y', False, False, False, False]

        out_dict= {}
        for n, nd, out, err, tf1, tf2 in zip(names, dnames, popt, perr, TF1, TF2):
            if tf1:
                out, err = np.array([out, err]) * self.umPerPixel
                if type(tf2) == str and crop_pixels_around_peak:
                    if tf2 == 'x':
                        out += self.maxCoors[0]
                    elif tf2 == 'y':
                        out += self.maxCoors[1]

            print ("{}\t\t{:2.2f} +/- {:1.2f}".format(n, out, err))
            out_dict[nd] = [out, err]
        



        
        return out_dict
        # return np.array([popt[3] * self.umPerPixel, popt[4] * self.umPerPixel])
    
    def fwhm_gaussian(self, w):
        return 2 * (2 * np.log(2))**0.5 * w
        

### Example run script    
if __name__ == "__main__":

    # folderPath = "/Volumes/GoogleDrive/My Drive/2019_Lund/Pre_plasma_diagnositc_calibration/"

    # imagefilepath = folderPath  + "Far_field.tif"

    # fs = focal_spot(image, plot_raw_input = False)    
    # # fs.Peaklocator(True)

    # umPerPixel = 2.575e-01
    # fit = fs.fit_2DGaus(umPerPixel, crop_pixels_around_peak = 250)
    # fs.createPlotWithLineOuts()

    # print ("Dictionary of fit params")
    # print (fit)
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-15/"
    run = "0001/"
    diagnostic = "Farfield pre/"
    # Load all the shot data
    folder_path = path_to_data + date + run + diagnostic
    filelist = func.FilesInFolder(folder_path, ".tif")
    shots = func.SplitArr(filelist, "_", 1)
    # Check that it is in order
    filelist, shots = func.sortArrAbyB(filelist, shots)
    
    # Calibration details
    umPerPixel = 2.575e-01 * 0.25
    
    out_dictionary = {}
    for f in filelist[:]:
        
        print (f)
        shot = f.split("_")[1]
        filepath = folder_path + f
        try:
            fs = FarField.focal_spot(filepath, plot_raw_input = False)  
            fit = fs.fit_2DGaus(umPerPixel, crop_pixels_around_peak = 250)
        except ValueError:
            print ("Fitting has gone wrong")
            fit = {'amp': [np.nan, np.nan],
                 'xc': [np.nan, np.nan],
                 'yc': [np.nan, np.nan],
                 'sigma_x': [np.nan, np.nan],
                 'sigma_y': [np.nan, np.nan],
                 'theta': [np.nan, np.nan],
                 'offset': [np.nan, np.nan]}
        out_dictionary[shot] = fit
    func.saveDictionary(path_to_data + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
                        out_dictionary)
        

