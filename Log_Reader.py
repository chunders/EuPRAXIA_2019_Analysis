#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Nov 18 16:07:45 2019

@author: chrisunderwood
    Log file reader.
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

import Functions3 as func
import pandas as pd

class logreader():
    def __init__(self, path, date, run):
        self.date = date
        self.run = run
        log_file_path = path + self.date +  self.date[:-1] + ".log"
        
        # Open in pandas
        self.df = pd.read_csv(log_file_path,
                         sep='\t')
        keys = self.df.keys()
        print (keys)
        
    def search_col(self, colname, operator, condition):
        if operator == "==":
            return self.df[self.df['colname']==condition].index.tolist()
        
    def replace_commas_with_dots(self, colname):
        col = self.df[colname]
        for i, d in enumerate(col):
            if type(d)==str:
                d = d.replace(',', '.')
                col.iloc[i] = d
        self.df[colname] = col
        
        print (self.df[colname])
        
                
    def return_not_nans(self, colname):
        return [i for i in self.df[colname] if type(i) is not float]
        
        
        
        

if __name__ == "__main__":
    path_to_data = "/Volumes/Lund_York/"
    date = "2019-11-15/"
    run = "0001/"

    r = logreader(path_to_data, date, run)
    # r.replace_commas_with_dots('Jet1x')
    df = r.df
    
    pressure_scan_shots = {}
    pressures = range(75, 400, 25)
    for pressure in pressures:
        indexes = df[df['CellPressure']==pressure].index.tolist()
        pressure_scan_shots[pressure] = indexes
        
        
    func.printDictionary(pressure_scan_shots)
