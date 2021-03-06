import numpy as np
from scipy.io import loadmat
from imageio import imread



def img2spec(filePath,calFile= r'Z:\2019 EuPRAXIA\OrielMysteryGrating_lambda.mat'):
    img = imread(filePath)
    specCal = loadmat(calFile)
    return specCal['lambdaAxis'].flatten(), np.mean(img.astype(float),axis=0)

