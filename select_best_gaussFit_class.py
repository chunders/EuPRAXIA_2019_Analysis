#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Jun 12 10:18:09 2018

@author: chrisunderwood
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class select_Gaus_fit():
    def __init__(self, x, y, plottingOnOff, guess):
        self.x = x
        self.y = y
        self.plotting = plottingOnOff
        self.guess = guess
        self.bounds = ( [-np.inf,-np.inf, 0, -np.inf],[np.inf,np.inf, np.inf,np.inf])
        if self.plotting:
            self.plotInput()
            
        self.fit_gaus_2_to_8()
        
    def gaus2(self, x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((x-x0)/c)**2) + y0

    def gaus4(self, x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((self.x-x0)/c)**4) + y0
    
    def gaus6(self, x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((self.x-x0)/c)**6) + y0
    
    def gaus8(self, x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((self.x-x0)/c)**8) + y0
    
    def plotInput(self):
        plt.plot(self.x,self.y)
        
    def fitGaus2(self):
        try:
            self.popt_G2, self.pcov_G2 = curve_fit(self.gaus2, self.x, self.y , 
                                                   p0=self.guess, bounds = self.bounds)
            self.fitG2 = self.gaus2(self.x, *self.popt_G2)
            
            if self.plotting:
                print('Fitting SG2: ')
                print(self.guess)
                print(self.popt_G2)
                print()
                plt.plot(self.x, self.fitG2, label = 'Fit G2')
        except:
            self.popt_G2 = [1,1,1,1]
            self.pcov_G2 = None
            self.fitG2  = np.ones(len(self.x))
            
    def fitGaus4(self):
        try:
            self.popt_G4, self.pcov_G4 = curve_fit(self.gaus4, self.x, self.y , 
                                                   p0=self.guess, bounds = self.bounds)
            self.fitG4 = self.gaus4(self.x, *self.popt_G4)        
            if self.plotting:
                print('Fitting SG4: ')
                print(self.guess)
                print(self.popt_G4)
                print()
                plt.plot(self.x, self.fitG4, label = 'Fit G4')
        except:
            self.popt_G4 = [1,1,1,1]
            self.pcov_G4 = None
            self.fitG4  = np.ones(len(self.x))
       
    def fitGaus6(self):
        try:
            self.popt_G6, self.pcov_G6 = curve_fit(self.gaus6, self.x, self.y , 
                                                   p0=self.guess, bounds = self.bounds)
            self.fitG6 = self.gaus6(self.x, *self.popt_G6)
            if self.plotting:
                print('Fitting SG6: ')
                print(self.guess)
                print(self.popt_G6)
                print()
                plt.plot(self.x, self.fitG6, label = 'Fit G6')    
        except:
            self.popt_G6 = [1,1,1,1]
            self.pcov_G6 = None
            self.fitG6  = np.ones(len(self.x))

    def fitGaus8(self):
        try:
            
            self.popt_G8, self.pcov_G8 = curve_fit(self.gaus8, self.x, self.y , 
                                                   p0=self.guess, bounds = self.bounds)
            self.fitG8 = self.gaus8(self.x, *self.popt_G8)
            if self.plotting:
                print('Fitting SG8: ')
                print(self.guess)
                print(self.popt_G8)
                print()
                plt.plot(self.x, self.fitG8, label = 'Fit G8')   
        except:
            self.popt_G8 = [1,1,1,1]
            self.pcov_G8 = None
            self.fitG8  = np.ones(len(self.x))
            
    def fit_gaus_2_to_8(self):
        self.fitOptions = ['SG2', 'SG4', 'SG6', 'SG8']
        self.fitPowers = [2, 4, 6, 8]

        self.fitGaus2()
        self.fitGaus4()
        self.fitGaus6()
        self.fitGaus8()
        self.fitParams = [self.popt_G2, self.popt_G4, self.popt_G6, self.popt_G8]
        self.fitErrors = [self.pcov_G2, self.pcov_G4, self.pcov_G6, self.pcov_G8]
        self.CalcR_2_value()

    def nearposn(self, array,value):
        posn = (abs(array-value)).argmin()
        return posn

        
    def CalcR_2_value(self):
        self.rrValues = []
        for fit in [self.fitG2, self.fitG4, self.fitG6, self.fitG8]:
            
            residuals = self.y - fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.y-np.mean(self.y))**2)
            self.rrValues.append( 1 - (ss_res / ss_tot) )
        maxRR = max(self.rrValues)
        self.bestFit = self.nearposn(self.rrValues, maxRR)
        outStr = ''
        CSI="\x1B[31;40m"
        CEND = '\x1B[0m'
        for i in self.rrValues:
            if i == self.rrValues[self.bestFit]:
                outStr += CSI + str(i) + ' ' + CEND
            else:
                outStr += str(i) + ' '
        if self.plotting: print(outStr) 
        
    def output(self):
        if self.plotting:
            print(self.fitOptions[self.bestFit])
            print(self.fitParams[self.bestFit])
            print(self.fitPowers[self.bestFit])
        return self.fitPowers[self.bestFit] , self.fitParams[self.bestFit]
    
    def getPopt1(self):
        return self.fitParams[self.bestFit][1]
        

    def BestFit(self, plotting = True, color = 'r', lw = 1, ax = None):
        order = self.fitPowers[self.bestFit]
        bestParams = self.fitParams[self.bestFit]
        # bestErrors = self.fitErrors[self.bestFit] #  All seem to be inf
        

        fit = 0
        if order == 2:
            fit =  self.gaus2(self.x, *bestParams)
        elif order == 4:
            fit = self.gaus4(self.x, *bestParams)

        elif order == 6:
            fit = self.gaus6(self.x, *bestParams)

        elif order == 8:  
            fit = self.gaus8(self.x, *bestParams)        
             
        if plotting:
            if ax == None:
                ax = plt.gca()
            ax.plot(self.x, fit, c = color, lw = lw, label = 'BestFit')
        return fit, order, bestParams

    def fwhm_of_best(self):
        best = self.BestFit(plotting = False)
        power = best[1]
        popt = best[2]
        print (popt, power)
        fwhm = 2 * popt[2]  * (np.log(2))**(1/(power)) 
        print (fwhm)
        return fwhm


if __name__ == "__main__":
    

    def gaus6( x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((x-x0)/c)**6) + y0
    def gaus2( x, *params):
        #Gaussian function
        A = params[0]
        x0 = params[1]
        c = params[2]
        y0 = params[3]
        return A*np.exp(-((x-x0)/c)**2) + y0    
    
    def manual_fwhm(x, fit, xc, y0):
        y = fit - y0
        xind_c = func.nearposn(x, xc)
        ind1 = func.nearposn(y[:xind_c], y.max()/2 )
        ind2 = func.nearposn(y[xind_c:], y.max()/2 ) + xind_c
        plt.plot(x[[ind1, ind2]], fit[[ind1, ind2]])
        fwhm = x[ind2] - x[ind1]
        print ("Manual fwhm", fwhm)
        return fwhm
    
    x = np.linspace(0, 100, 1e4)
    y = gaus2(x, *[10, 50, 30, 10]) # + np.random.rand(len(x))
    gf = select_Gaus_fit(x,y,False, [5, 55, 32, 2])
    plt.show()
    out = gf.BestFit(plotting = False)
    plt.plot(x,y)
    plt.plot(x, out[0])
    
    fwhm_params = gf.fwhm_of_best()
    fwhmMan = manual_fwhm(x, out[0], out[2][2], out[2][3])
    
    print (fwhm_params / fwhmMan)
    # print (out[1])
