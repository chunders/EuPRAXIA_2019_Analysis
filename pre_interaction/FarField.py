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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

import sys
path_to_git = "/Volumes/GoogleDrive/My Drive/2019_Lund/EuPRAXIA_2019_Analysis/"
sys.path.append( path_to_git )
import Functions3 as func
from scipy.signal import peak_prominences


imageRead = True
if imageRead:
    from skimage import io
else:
    from PIL import Image

class focal_spot(): 
    #Takes the folder path and then the file name
    def __init__(self, filepath, calculations = True, simpleVersion =  True, 
                 backgroundImage = None, plot_raw_input = True, 
                 plotCalcs = False):
        # The image should be a numpy array
        self.filepath = filepath
        self.load_image(backgroundImage)
        if plot_raw_input:
            r = self.im.max() - self.im.min()
            plt.imshow(self.im, vmax =  self.im.min() + 0.4* r )
            plt.colorbar()
            plt.show()
                    
        if calculations:
            self.create_class_variables(simpleVersion, plotting = plotCalcs )
            

    
    def load_image(self, backgroundImage):
        self.im = io.imread(self.filepath)    
        if backgroundImage is not None:
            # Check that the image is a float before taking the background away
            self.im = np.float64(self.im) -  np.float64(np.array(backgroundImage))

                        
    def create_class_variables(self, simpleVersion =  True, plotting = True):
        self.imShape = np.shape(self.im)
        self.background_subtraction()
        self.normalise()
#        self.maxCoors = np.unravel_index(self.im.argmax(), self.im.shape)
        self.Peaklocator(plotting = plotting, simpleVersion =  simpleVersion )
        self.ThresholdImage()
        self.lineOutIntegrals()
        
    def background_subtraction(self):
        # Basic background subtraction, zero the lowest value
        self.im = self.im - self.im.min()

    def normalise(self):
        FS_im_bg = (self.im - self.im.min())
        self.im_bg_n = (FS_im_bg * 1.0) / FS_im_bg.max()        
        
    def Peaklocator(self, plotting = False, simpleVersion = True):
        self.xlineout = self.im.sum(axis = 0)
        self.ylineout = self.im.sum(axis = 1)     
        if plotting and False:
            plt.plot(self.xlineout, label = 'x')
            plt.plot(self.ylineout, label = 'y')
            plt.legend()
            plt.show()        
        
  
      # Find the max coors
        if simpleVersion:
            self.maxCoors = [self.xlineout.tolist().index(max(self.xlineout)), 
                         self.ylineout.tolist().index(max(self.ylineout))]
        else:
            try:
                # Fit to work out best position
                x = np.arange(len(self.xlineout))
                xg = [self.xlineout.max(), len(self.xlineout) / 2, len(self.xlineout) / 4, self.xlineout.min()]
                yg = [self.ylineout.max(), len(self.ylineout) / 2, len(self.ylineout) / 4, self.ylineout.min()]
                poptx, _ = curve_fit(func.gaus, x, self.xlineout, 
                                     p0 = xg,
                                      bounds = ([1e-9, 0, 1e-9, -np.inf],  
                                                [np.inf, len(self.ylineout), np.inf, np.inf])                                     
                                     )
                print ('poptx ',poptx)
                
                y = np.arange(len(self.ylineout))
                popty, _ = curve_fit(func.gaus, y, self.ylineout, 
                                     p0 = yg,
                                      bounds = ([1e-9, 0, 1e-9, -np.inf],  
                                                [np.inf, len(self.ylineout), np.inf, np.inf])
                                     ) 
                print ('popty ',popty)                
                self.maxCoors = [int(np.rint(poptx[1])), int(np.rint(popty[1]))]
                if self.maxCoors[0] >= self.im.shape[1]:
                    self.maxCoors[0] = self.im.shape[1] - 1
                if self.maxCoors[1] >= self.im.shape[0]:
                    self.maxCoors[1] = self.im.shape[0] - 1                    
                # peak =  [poptx[0], popty[0]]
            except RuntimeError:
                print ("Run time error")
                print (xg, yg)
                # print (poptx, popty)
                assert False, "The peaks can't be found"
            
            
            
        print("Maximum Coordinate (pixel nos): ", self.maxCoors)
        if plotting:
            plt.plot(range(len(self.ylineout)), self.ylineout)
            plt.plot(range(len(self.xlineout)), self.xlineout)     
            if not simpleVersion:
                plt.plot(self.maxCoors[0], self.xlineout.max(), 's')
                plt.plot(self.maxCoors[1], self.ylineout.max(), 's')            
            plt.show()

        print (self.maxCoors)
        # print (int(self.maxCoors[0]), int(self.maxCoors[1]))
        x_prom = self.xlineout[int(self.maxCoors[0]) ] / np.average(self.xlineout)
        y_prom = self.ylineout[int(self.maxCoors[1]) ] / np.average(self.ylineout)
        print (x_prom)
        print (y_prom)
        # if not x_prom[0][0] > 0.7:
        #      print("xpeak not promient", x_prom[0][0])
        # if not y_prom[0][0] > 0.7:
        #      print("ypeak not promient", y_prom[0][0])            
        assert x_prom > 2, print("xpeak not promient", x_prom)
        assert y_prom > 2, print("ypeak not promient", y_prom)
            
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
            
        
            
        print ('In fit 2d gaus')
        print (crop_pixels_around_peak)
        print ('shape', shape, 'max coors', maxCoors)


        self.umPerPixel = umPerPixel
        
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        x, y = np.meshgrid(x, y)
        if guessWhole:
            #               amplitude, xo, yo 
            initial_guess = [image.max(), maxCoors[1], maxCoors[0],
                         # sigma_x, sigma_y, theta, offset
                         50, 50, 0, 0]
        else:
            initial_guess = [image.max(), crop_pixels_around_peak, crop_pixels_around_peak,
                         # sigma_x, sigma_y, theta, offset
                         50, 50, 0, 0]           
        print (initial_guess )
            
        bounds = ((1e-9,np.inf), # amplitude
                   (0, np.inf), # xo
                   (0, np.inf), # xo
                   (1e-9,np.inf), # sigma_x
                   (1e-9,np.inf), # sigma_y
                   (-np.inf,np.inf), # theta
                   (-np.inf,np.inf)  # offset
                   )
        bounds = (np.ones(7) * -np.inf, np.ones(7) * np.inf)
        bounds = np.array(bounds).reshape((2,7))
        print ("Pre Fitting")
        popt, pcov = curve_fit(self.twoD_Gaussian, (x, y), image.T.ravel(), p0=initial_guess,                                        
                                  bounds = bounds
                                     )
        print ("Post Fitting")        
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
                    # Should this be * self.umPerPixel
                    # maxCoors is in pixels.
                    if tf2 == 'x':
                        out += self.maxCoors[0] * self.umPerPixel
                    elif tf2 == 'y':
                        out += self.maxCoors[1] * self.umPerPixel

            print ("{}\t\t{:2.2f} +/- {:1.2f}".format(n, out, err))
            out_dict[nd] = [out, err]
        



        
        return out_dict
        # return np.array([popt[3] * self.umPerPixel, popt[4] * self.umPerPixel])
    
    def fwhm_gaussian(self, w):
        return 2 * (2 * np.log(2))**0.5 * w
        

### Example run script    
if __name__ == "__main__":

    path_to_data = "/Volumes/CIDU_passport/2019_Lund_Data/"
    savePath = "/Volumes/CIDU_passport/2019_Lund_Analysis/"
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
    for f in filelist[5:7]:
        
        print (f)
        shot = f.split("_")[1]
        filepath = folder_path + f

        try:            
            fs = focal_spot(filepath, plot_raw_input = True, simpleVersion = False)  
            nextStep = True
        except AssertionError:
            print ("The peaks are not prominent -> No focal spot")
            fit = {'amp': [np.nan, np.nan],
                 'xc': [np.nan, np.nan],
                 'yc': [np.nan, np.nan],
                 'sigma_x': [np.nan, np.nan],
                 'sigma_y': [np.nan, np.nan],
                 'theta': [np.nan, np.nan],
                 'offset': [np.nan, np.nan]}   
            nextStep = False                
        if nextStep:
            try:
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
                plt.imshow(fs.im, 
                           # norm = mpl.colors.LogNorm() 
                           )
                plt.plot(fs.maxCoors[0], fs.maxCoors[1], 'rx', markersize = 15)
                plt.colorbar()
                plt.show()
            
                plt.imshow(fs.im, 
                           # norm = mpl.colors.LogNorm() 
                           )
                plt.plot(fs.maxCoors[0], fs.maxCoors[1], 'rx', markersize = 15)
                plt.colorbar()            
                plt.show()
            
        out_dictionary[shot] = fit
        print ('Completed', f, '\n')
 
    # func.saveDictionary(savePath + date + run + diagnostic[:-1].replace(" ", "_") + "_extraction.json",
    #                     out_dictionary)
        

