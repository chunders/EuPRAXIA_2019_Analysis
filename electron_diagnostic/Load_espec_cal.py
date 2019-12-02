#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Sun Dec  1 12:20:52 2019

@author: chrisunderwood
"""
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, "..")
import Functions3 as func


def lin(x, *params):
    m = params[0]
    c = params[1]
    return m * x + c

# folderPath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\'
folderPath = "/Volumes/Lund_York/Electron_spectrometer/Inputs/"
filePath = folderPath + "especCalib_EuPRAXIA.txt"
calFile = np.loadtxt(filePath)
distance = calFile[:,0]
distance_mm = distance * 1e3
Energy = calFile[:,1]
travel_Len = calFile[:,2]
travel_Len_mm = travel_Len * 1e3
sizeOfImage = 2048

# =============================================================================
# Position of the pixels wrt lanex positions
# =============================================================================
pos_m = [0.05, 0.1, 0.15, 0.2]
pos_mm = np.array(pos_m) * 1e3
pixels = [320, 728, 1148, 1578]
pixToPos = np.c_[pixels, pos_mm]



# =============================================================================
# # Linear fit to get pixel positions on the lanex
# =============================================================================
xpixel = np.arange(sizeOfImage)
popt, pcov = curve_fit(lin, pixToPos[:,0], pixToPos[:,1], p0 = [1, 0])
xDist_mm = lin(xpixel, *popt)

# =============================================================================
# # Extrapolate the distance and energy curve
# =============================================================================
f = interp1d(distance_mm, Energy, kind = 'cubic', bounds_error = False,
        fill_value='extrapolate')
energyPerPixel_MeV_interp1d = f(xDist_mm)

# =============================================================================
# # Work out the angular size of each pixel
# =============================================================================
pixSize_mm = xDist_mm[1] - xDist_mm[0]
angularSizePix_rads = np.arctan( pixSize_mm / travel_Len_mm )

# interpolate to get distance traveled per pixel
angularSizePix_mrads = angularSizePix_rads * 1e3
travelPerPix = interp1d(distance_mm, travel_Len_mm, kind = 'cubic', bounds_error = False,
        fill_value='extrapolate')
pathLenToPix = travelPerPix(xDist_mm)

cal_divergence = np.tan(pixSize_mm / travel_Len_mm) * 1e3


angularSizeOfPixel = np.tan(pixSize_mm / pathLenToPix)
angularSizeOfPixel_mrads = angularSizeOfPixel * 1e3


pix_Calibration = []
for x in distance_mm:
    ind = func.nearposn(xDist_mm, x)
    pix_Calibration.append(ind)
''' 
plt.plot(xpixel, pathLenToPix, label = 'fit')
plt.plot(pix_Calibration, travel_Len_mm, label = 'Cal')
plt.legend()
plt.show()


plt.plot(xpixel, angularSizeOfPixel_mrads)

'''
f, ax = plt.subplots(nrows = 3, sharex = True)
# Show the pixels and distance
ax[0].set_title('Pixel to Distance, calibration points')

ax[0].set_ylabel("Distance\n(mm)")
# ax[0].set_xlabel('Pixel Nos')

ax[0].plot(pixToPos[:,0], pixToPos[:,1], 's-', label = "Cal Points")
ax[0].plot(xpixel, xDist_mm, label = "lin fit")
ax[0].legend()

ax[1].plot(pix_Calibration, cal_divergence, 's-r', label = "Cal Points")
ax[1].plot(xpixel, angularSizeOfPixel_mrads)
# ax1 = func.TwinAxis(ax[1], LHSCol = 'r', RHSCol = 'b')
# ax1.plot(distance_mm, angularSizePix_mrads, 'b')
ax[1].legend()
ax[1].set_ylabel("Angular pixel size\n(mrads)")


ax[2].plot(pix_Calibration, Energy, 's-r', label = "Cal Points")
ax[2].plot(xpixel, energyPerPixel_MeV_interp1d, label = "Fit 1D interp")
ax[2].legend()
ax[2].set_ylim([0, 400])

## Testing the two different intepolation methods against each other.
# for order in range(1, 4):
#     f = InterpolatedUnivariateSpline(distance_mm, Energy, k=order)
#     energyPerPixel_MeV = f(xDist_mm)

#     ax[2].plot(xpixel, energyPerPixel_MeV - energyPerPixel_MeV_interp1d, label = order)
#     ax[2].legend()

ax[2].set_xlabel('Pixel Nos')
ax[2].set_ylabel('Energy\n(MeV)')

savePath = folderPath + 'especCalib_per_pix.txt'    
np.savetxt(savePath, np.c_[xpixel, xDist_mm, energyPerPixel_MeV_interp1d, angularSizeOfPixel_mrads])


# savePath = folderPath + 'especCalib_dist_per_pix.txt'    
# np.savetxt(savePath, xDist_mm)
# print (energyPerPixel_MeV)

# print (calFile)
# plt.ion()
# plt.plot(calFile[:,0], calFile[:,1])
# plt.show()

# xpixel = np.arange(sizeOfImage)
# distance = calFile[:,0]
# Energy = calFile[:,1]

# # print(distance)

# pixelsPos = distance/distance.max() * sizeOfImage
# f = interp1d(pixelsPos, Energy, kind = 'cubic')
# xEnergy = f(xpixel)
# print (pixelsPos)


filePath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_EuPRAXIA.txt'

plt.tight_layout()
plt.show()

