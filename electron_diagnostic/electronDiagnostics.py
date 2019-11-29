import numpy as np
from scipy.io import loadmat
from imageio import imread
from scipy.ndimage.filters import median_filter as mf
from scipy.interpolate import interp1d
pC_per_count = 1.5585e-7


def load_Ecalibration():
    # filePath = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_EuPRAXIA.txt'
    # calFile = np.loadtxt(filePath)
    # sizeOfImage = 2048
    # xpixel = np.arange(sizeOfImage) # i've assumed you will change this variable to x in mm and that it is linear
    # distance = calFile[:,0]
    # Energy = calFile[:,1]
    # pixelsPos = distance/distance.max() * sizeOfImage
    # f = interp1d(pixelsPos, Energy, kind = 'cubic')
    # xEnergy = f(xpixel) 
    espec_cal_per_pix_file = r'Z:\2019 EuPRAXIA\Electron_spectrometer\Inputs\especCalib_energyMeV_per_pix.txt'    
    calibrationFile = np.loadtxt(espec_cal_per_pix_file)

    xpixel = calibrationFile[:,0]
    xEnergy = calibrationFile[:,1]

    dEdx = np.abs(np.gradient(xEnergy.flatten())/np.gradient(xpixel.flatten()))
    eAxis_MeV = np.linspace(np.min(xEnergy),np.max(xEnergy),num=1000,endpoint=True)
    return xEnergy, dEdx, eAxis_MeV, xpixel

def loadLanex(filePath):
    img = imread(filePath)
    return img

def im2lineout(img):
    img = mf(img,3)
    img = img - np.median(img)
    img = img.astype(float)*pC_per_count
    return np.sum(img,axis=0)

def getLaxexLineouts(filePaths):
    lineoutList = []
    for f in filePaths:
        img = loadLanex(f)
        lineoutList.append(im2lineout(img))
    return lineoutList



def img2spec(filePath):
    xEnergy, dEdx, eAxis_MeV, x_mm = load_Ecalibration()
    img = loadLanex(filePath)
    print(np.shape(img))
    dEdx_lin = im2lineout(img)
    lineout_pC_per_mm = dEdx_lin/np.abs(np.gradient(x_mm.flatten()))
    specFunc = interp1d(xEnergy,lineout_pC_per_mm/dEdx,bounds_error=None, fill_value=0)
    
    spec_pC_per_MeV = specFunc(eAxis_MeV)
    return eAxis_MeV, spec_pC_per_MeV


def getLaxexSpectra(filePaths):
    specList = []
    for f in filePaths:
        eAxis_MeV, spec_pC_per_MeV = img2spec(f)
        specList.append(spec_pC_per_MeV)
    return eAxis_MeV, specList