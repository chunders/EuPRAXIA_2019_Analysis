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

def im2Restrictedlineout(img,lL,uL):
    # For use when only looking at spectrum of electrons 
    # close to the axis
    img = mf(img,3)
    img = img - np.median(img)
    img = img.astype(float)*pC_per_count
    img = img[lL:uL,:]
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
    # print(np.shape(img))
    dEdx_lin = im2lineout(img)
    lineout_pC_per_mm = dEdx_lin/np.abs(np.gradient(x_mm.flatten()))
    specFunc = interp1d(xEnergy,lineout_pC_per_mm/dEdx,bounds_error=None, fill_value=0)
    
    spec_pC_per_MeV = specFunc(eAxis_MeV)
    return eAxis_MeV, spec_pC_per_MeV


def img2Restrictedspec(filePath,lL,uL):
    # For use when only looking at spectrum of electrons 
    # close to the axis
    xEnergy, dEdx, eAxis_MeV, x_mm = load_Ecalibration()
    img = loadLanex(filePath)
    # print(np.shape(img))
    dEdx_lin = im2Restrictedlineout(img,lL,uL)
    lineout_pC_per_mm = dEdx_lin/np.abs(np.gradient(x_mm.flatten()))
    specFunc = interp1d(xEnergy,lineout_pC_per_mm/dEdx,bounds_error=None, fill_value=0)
    
    spec_pC_per_MeV = specFunc(eAxis_MeV)
    return eAxis_MeV, spec_pC_per_MeV


def getLaxexSpectra(filePaths, file_dictionary = None):
    # Use a dictionary to check whether the data has been extracted before
    if file_dictionary == None:
        print ('Creating blank dictionary')
        file_dictionary = {}


    specList = []
    for f in filePaths:
        if f in file_dictionary.keys():
            specList.append(file_dictionary[f])
        else:
            try:
                eAxis_MeV, spec_pC_per_MeV = img2spec(f)
                specList.append(spec_pC_per_MeV)
                file_dictionary[f] = spec_pC_per_MeV
            except ValueError:
                # Exception to catch when the size of the arrays have changed
                # 
                print ('Bad File', f)
                file_dictionary[f] = np.zeros(1000) 


    if not 'eAxis_MeV' in locals():
        print ('Making energy axis for lanex')
        eAxis_MeV, spec_pC_per_MeV = img2spec(filePaths[0])


    return eAxis_MeV, specList, file_dictionary

def getRestrictedLaxexSpectra(filePaths, lL,uL, file_dictionary = None):
    # Use a dictionary to check whether the data has been extracted before
    if file_dictionary == None:
        print ('Creating blank dictionary')
        file_dictionary = {}


    specList = []
    for f in filePaths:
        if f in file_dictionary.keys():
            specList.append(file_dictionary[f])
        else:
            try:
                eAxis_MeV, spec_pC_per_MeV = img2Restrictedspec(f,lL,uL)
                specList.append(spec_pC_per_MeV)
                file_dictionary[f] = spec_pC_per_MeV
            except ValueError:
                # Exception to catch when the size of the arrays have changed
                # 
                print ('Bad File', f)
                file_dictionary[f] = np.zeros(1000) 


    if not 'eAxis_MeV' in locals():
        print ('Making energy axis for lanex')
        eAxis_MeV, spec_pC_per_MeV = img2Restrictedspec(filePaths[0],lL,uL)


    return eAxis_MeV, specList, file_dictionary


def getLaxexSpectra_andBADfiles(filePaths, file_dictionary = None):
    # Use a dictionary to check whether the data has been extracted before
    if file_dictionary == None:
        print ('Creating blank dictionary')
        file_dictionary = {}

    badFiles = []
    specList = []
    for f in filePaths:
        if f in file_dictionary.keys():
            specList.append(file_dictionary[f])
        else:
            try:
                eAxis_MeV, spec_pC_per_MeV = img2spec(f)
                specList.append(spec_pC_per_MeV)
                file_dictionary[f] = spec_pC_per_MeV
            except ValueError:
                # Exception to catch when the size of the arrays have changed
                # 
                print ('Bad File', f)
                badFiles.append(f)
                file_dictionary[f] = np.zeros(1000) 


    if not 'eAxis_MeV' in locals():
        print ('Making energy axis for lanex')
        eAxis_MeV, spec_pC_per_MeV = img2spec(filePaths[0])


    return eAxis_MeV, specList, file_dictionary, badFiles


