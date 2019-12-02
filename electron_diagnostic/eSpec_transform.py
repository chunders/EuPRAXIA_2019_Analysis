#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Sun Dec  1 18:25:53 2019

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
from skimage.io import imread

pC_per_count = 1.5585e-7


def imshow_with_lineouts(image, fitlerSize = 3, CropY=None,
                         x = [], y = [], xlabel = '', ylabel = '',
                         aspect='auto', ylim = None,
                         **kwargs):
    import matplotlib.gridspec as gridspec
    from scipy.signal import medfilt
    
        #Sum in each direction for lineouts
    if len(x) == 0:
        x = np.arange(image.shape[1])
    if len(y) == 0:
        y = np.arange(image.shape[0])    
    sumX = np.nansum(image,axis = 0)
    sumY = np.nansum(image,axis = 1)

    fig = plt.figure(figsize=(8,8))
    # 3 Plots with one major one and two extra ones for each axes.
    gs = gridspec.GridSpec(4, 4, height_ratios=(1,1,1,1), 
                           width_ratios=(0.5,1,1,1))
    gs.update(wspace=0.025, hspace=0.025)
    
    #    Create all axis, including an additional one for the cbar
    ax_main = plt.subplot(gs[0:3, 1:-1])             # Image
    ax_main.axis('off')
    ax_right = plt.subplot(gs[0:3, 0] ) # right hand side plot
    ax_below = plt.subplot(gs[-1, 1:-1] ) # below plot
    
    cax4 = fig.add_axes([0.7, 0.35, 0.05, 0.5])
    
    im = ax_main.imshow(image, aspect=aspect ,**kwargs )
    ax_below.plot(x, medfilt(sumX, fitlerSize))
    ax_right.plot(medfilt(sumY, fitlerSize), y)
    ax_below.set_xlabel(xlabel)
    ax_right.set_ylabel(ylabel)
    func.scientificAxis('y', ax = ax_below)
    func.scientificAxis('x', ax = ax_right)    
    try:
        ax_right.set_ylim([-ylim, ylim])
        low = func.nearposn(y, -ylim)
        high = func.nearposn(y, ylim)        
        ax_main.set_ylim([low, high])
    except TypeError:
        pass
    
    plt.colorbar(im, cax = cax4)




def load_calibration():
    filePath = 'especCalib_per_pix.txt'
    calFile = np.loadtxt(filePath)
    xPixNo =    np.array(calFile[:,0], dtype = int)
    xDist_mm =  calFile[:,1]
    energyPerPix_MeV = calFile[:,2]
    angularSizeOfPixel_mrads = calFile[:,3]
    return xPixNo, xDist_mm, energyPerPix_MeV, angularSizeOfPixel_mrads

def loadImage(height = 543):
    start = 867
    im = np.zeros((height, 2048))   
    # for i in range(1, 2):
    #     file = "/Volumes/Lund_York/2019-11-26/0002/Lanex/0002_000{}_Lanex.tif".format(i)
    #     d = imread(file)
    #     d = np.array(d[start:start + height,:])
    #     im += d
    file = "/Volumes/Lund_York/2019-11-26/0002/Lanex/0002_0010_Lanex.tif"
    d = imread(file)
    d = np.array(d[start:start + height,:])
    im = d

    return im * pC_per_count

def closest_argmin(search, mainArr):
    index=[]
    for point in search:
        index.append(func.nearposn(mainArr, point))
    return index
    
def evenlySpacedDivergence(div, start, end, xPixNo, nbins = 20):
    binEdges = np.linspace(start, end, num = nbins,
                             endpoint= True)
    indexes = closest_argmin(binEdges, div)
    xCenters = xPixNo[indexes]
    return binEdges, xCenters, indexes


def evenlySpacedEnergy(EPerPix_MeV, xPixNo, nbins = 200):
    binEdges = np.linspace(EPerPix_MeV[0], EPerPix_MeV[-1], num = nbins,
                             endpoint= True)
    indexes = closest_argmin(binEdges, EPerPix_MeV)
    xCenters = xPixNo[indexes]
    return binEdges, xCenters, indexes

def histgramPoints(arr, binEdges):
    hist, bin_edges = np.histogram(arr, bins = binEdges)
    binCenters = (bin_edges[1:] + bin_edges[:-1] )* 0.5
    
    return hist, bin_edges, binCenters
    

def centerBins(binEdges):
    binCenters = (bin_edges[1:] + bin_edges[:-1] )* 0.5
    return  binCenters
    
def createHistogram(arr, binEdges):
    hist = []
    for i in range(len(binEdges)- 1):
        step = abs(binEdges[i+1] - binEdges[i])
        # if step == 0:
        #     step = 1
        hist.append(np.sum(arr[binEdges[i]:binEdges[i+1]]) /step
                    )
    return hist

def rehist_2D(arr, binEdges):
    hist = []
    dE_list = []
    for i in range(len(binEdges)- 1):
        dE = abs(binEdges[i] - binEdges[i+1])
        dE_list.append(dE)
        hist.append( 
                arr[:, binEdges[i]:binEdges[i+1] ].sum(axis = 1) / dE 
                )
    hist = np.array(hist).T
    return hist, dE_list

def plot_calibration(xPixNo, EPerPix_MeV, xCenters, binEdges, mradsPerPix):
    ax = plt.subplot()
    ax.plot(xPixNo, EPerPix_MeV, 'r.-')    
    ax.plot(xCenters, binEdges)
    ax1 = func.TwinAxis(ax)
    ax1.plot(xPixNo, mradsPerPix, 'b.-')
    plt.show()
    plt.title("Number of pixels per bin")
    plt.plot(binCenters, hist_pointPerEnergy)
    plt.hlines(0, binCenters[0], binCenters[-1])
    plt.show()
    
    
if __name__ == "__main__":
    
    xPixNo, xDist, EPerPix_MeV, mradsPerPix = load_calibration()
    height = 543

    binEdges, xCenters, indexes = evenlySpacedEnergy(EPerPix_MeV, xPixNo,
                                                     nbins = 160
                                                     )

#     # binEdges = np.linspace(EPerPix_MeV[0], EPerPix_MeV[-1], num = 200,
#     #                          endpoint= True)
#     # indexes = closest_argmin(binEdges, EPerPix_MeV)
#     # xCenters = xPixNo[indexes]
    
    hist_pointPerEnergy, bin_edges, binCenters = histgramPoints(EPerPix_MeV, binEdges = binEdges)
    energy_bins = np.array(binCenters) * 1
    
    # plot_calibration(xPixNo, EPerPix_MeV, xCenters, binEdges, mradsPerPix)



    im = loadImage(height)
    # axis 0 is x
    # axis 1 is y
    # plt.imshow(im)
    # plt.colorbar()
    # plt.show()
    lineout = im.sum(axis=0)
    hist = createHistogram(lineout, indexes)
    hist2d, dElist = rehist_2D(im, indexes)
    l = hist2d.sum(axis=1)
    indCenter = func.nearposn(l, l.max())
    
    f, ax = plt.subplots(nrows = 2)
    ax[0].plot(lineout)
    ax[1].plot(binCenters, hist, 'r', label = "1D")
    # ax1 = func.TwinAxis(ax[1])
    ax[1].plot(binCenters, hist2d.sum(axis = 0), 'b', label = "2D")
    ax[1].legend(loc = 1)
    # ax1.legend(loc = 2)    
    plt.show()
    
    # ax[1].plot(binCenters, hist)
    ext = (binCenters[0],binCenters[-1],
           -(im.shape[0]) / 2, (im.shape[0] ) /2 )
    plt.imshow(hist2d, norm = mpl.colors.LogNorm(), 
               extent=ext, aspect = 'auto', origin = 'lower'
               )
    plt.xlabel("Energy")
    plt.ylabel("Pixels")
    plt.colorbar()
    plt.show()
    
    # plt.plot(dElist)
    # divHm = []
    # for rad in mradsPerPix:
    #     divHm.append( (xPixNo - len(xPixNo)//2) * rad)
    # plt.imshow(divHm)
    # plt.colorbar()
    
    if False:
        beamAxis = 241 # The pixel number that corresponds to the beam axis.
    else:
        l = hist2d.sum(axis = 1)
        beamAxis = func.nearposn(l, l.max())

    diverenceBins = 545
    divergenceRange = (height * mradsPerPix.max() // 2)
    newDivEdges = np.linspace(-divergenceRange, divergenceRange, num = diverenceBins)
    DivCenters = (newDivEdges[:-1] + newDivEdges[1:]) * 0.5
    new_image = []
    for i, row in enumerate(hist2d.T[:]):
        divPerPixelRow = (np.arange(543) - beamAxis) * mradsPerPix[i]
        binEdges, xCenters, indexes = evenlySpacedDivergence(divPerPixelRow, newDivEdges[0], newDivEdges[-1],
                                         xPixNo,nbins = diverenceBins)
        lineout = createHistogram(row, indexes)
        new_image.append(lineout[::-1])
        # # plt.plot(divPerPixelRow)
        # # plt.plot(row)
        # hist, edges = np.histogram(row,  bins = newDivEdges)
        # new_image.append(hist)
    new_image = np.array(new_image).T
    ext = (binCenters[0],binCenters[-1],
          newDivEdges[0], newDivEdges[-1] )
    
    plt.imshow(new_image, aspect = 'auto', extent=ext)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Div (mrads)")
    plt.colorbar()
    # plt.ylim([-15, 15])
    plt.show()
    
    imshow_with_lineouts(new_image, y = DivCenters, x = energy_bins, 
                         ylabel ='Div (mrads)', xlabel = 'Energy (MeV)', 
                         ylim = 30)
    




