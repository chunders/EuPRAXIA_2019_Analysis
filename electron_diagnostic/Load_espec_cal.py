import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def lin(x, *params):
    m = params[0]
    c = params[1]
    return m * x + c

filePath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_EuPRAXIA.txt'
calFile = np.loadtxt(filePath)
distance = calFile[:,0]
distance_mm = distance * 1e3
Energy = calFile[:,1]
sizeOfImage = 2048

pos_m = [0.05, 0.1, 0.15, 0.2]
pos_mm = np.array(pos_m) * 1e3
pixels = [320, 728, 1148, 1578]
pixToPos = np.c_[pixels, pos_mm]


f, ax = plt.subplots(nrows = 3)
# Show the pixels and distance
ax[0].set_title('Pixel to Distance, calibration points')
ax[0].plot(pixToPos[:,0], pixToPos[:,1], '.-')
ax[0].set_xlabel('Pixel Nos')


# Interpolate to get pixel positions
f = interp1d(pixToPos[:,0], pixToPos[:,1])
xpixel = np.arange(sizeOfImage)
popt, pcov = curve_fit(lin, pixToPos[:,0], pixToPos[:,1], p0 = [1, 0])
xDist_mm = lin(xpixel, *popt)
ax[1].set_ylabel("Distance (mm)")
ax[1].set_xlabel('Pixel Nos')
ax[1].plot(xpixel, xDist_mm)



print(distance[0] * 1e3, distance[-1]*1e3 )
print(xDist_mm[0], xDist_mm[-1])

f = interp1d(distance_mm, Energy, kind = 'cubic', bounds_error = False,
        fill_value='extrapolate')
energyPerPixel_MeV = f(xDist_mm)

ax[2].plot(xpixel, energyPerPixel_MeV, label = "Scipy")

ax[2].set_ylim([0, 400])

from scipy.interpolate import InterpolatedUnivariateSpline
for order in range(1, 4):
    f = InterpolatedUnivariateSpline(distance_mm, Energy, k=order)
    energyPerPixel_MeV = f(xDist_mm)

    ax[2].plot(xpixel, energyPerPixel_MeV, label = order)
    ax[2].legend()

ax[2].set_xlabel('Pixs')
ax[2].set_ylabel('Energy (MeV)')

savePath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_energyMeV_per_pix.txt'    
np.savetxt(savePath, np.c_[xDist_mm, energyPerPixel_MeV])


# savePath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_dist_per_pix.txt'    
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

