import numpy as np
from scipy.io import loadmat
from imageio import imread
calFile = r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda.mat'

specCal = loadmat(calFile)

def img2spec(filePath):
    img = imread(filePath)
    return specCal['lambdaAxis'], np.mean(img.astype(float),axis=0)
