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

    return im

def closest_argmin(search, mainArr):
    index=[]
    for point in search:
        index.append(func.nearposn(mainArr, point))
    return index
    
def evenlySpacedDivergence(div, start, end, nbins = 20):
    binEdges = np.linspace(start, end, num = nbins,
                             endpoint= True)
    indexes = closest_argmin(binEdges, div)
    xCenters = xPixNo[indexes]
    return binEdges, xCenters, indexes


def evenlySpacedEnergy(EPerPix_MeV, nbins = 200):
    binEdges = np.linspace(EPerPix_MeV[0], EPerPix_MeV[-1], num = nbins,
                             endpoint= True)
    indexes = closest_argmin(binEdges, EPerPix_MeV)
    xCenters = xPixNo[indexes]
    return binEdges, xCenters, indexes

def histgramPoints(arr, binEdges):
    hist, bin_edges = np.histogram(arr, bins = binEdges)
    binCenters = (bin_edges[1:] + bin_edges[:-1] )* 0.5
    
    return hist, bin_edges, binCenters
    
    
def createHistogram(arr, binEdges):
    hist = []
    for i in range(len(binEdges)- 1):
        step = abs(binEdges[i+1] - binEdges[i])
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

    binEdges, xCenters, indexes = evenlySpacedEnergy(EPerPix_MeV,
                                                     nbins = 160
                                                     )

#     # binEdges = np.linspace(EPerPix_MeV[0], EPerPix_MeV[-1], num = 200,
#     #                          endpoint= True)
#     # indexes = closest_argmin(binEdges, EPerPix_MeV)
#     # xCenters = xPixNo[indexes]
    
    hist_pointPerEnergy, bin_edges, binCenters = histgramPoints(EPerPix_MeV, binEdges = binEdges)
    

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

    diverenceBins = 550
    divergenceRange = (height * mradsPerPix.max() // 2)
    newDivEdges = np.linspace(-divergenceRange, divergenceRange, num = diverenceBins)
    DivCenters = (newDivEdges[:-1] + newDivEdges[1:]) * 0.5
    new_image = []
    for i, row in enumerate(hist2d.T[:]):
        divPerPixelRow = (np.arange(543) - beamAxis) * mradsPerPix[i]
        binEdges, xCenters, indexes = evenlySpacedDivergence(divPerPixelRow, newDivEdges[0], newDivEdges[-1],
                               nbins = diverenceBins)
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
    




