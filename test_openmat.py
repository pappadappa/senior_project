# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:43:31 2020

@author: Admin
"""
# import tables

# file = tables.openFile('f_FFT_Nor_106_2b1_Pl_mc_LittC2SE_1.mat')
# lon = file.root.lon[:]
# lat = file.root.lat[:]
# # Alternate syntax if the variable name is in a string
# varname = 'lon'
# lon = file.getNode('/' + varname)[:]

# from scipy.io import loadmat
# x = loadmat('1.mat')
# lon = x['lon']
# lat = x['lat']
# # one-liner to read a single variable
# lon = loadmat('test.mat')['lon']

# from os.path import dirname, join as pjoin
# import scipy.io as sio


# path = 

# data_dir = pjoin(dirname(path), 'matlab', 'tests', 'data')
# mat_fname = pjoin(data_dir, '1.mat')
# mat_contents = sio.loadmat(mat_fname)


# print(mat_contents['testdouble'])


#---------------------------------------
#open .mat succes

from scipy.io import loadmat
import pandas as pd
import os 
  
# change the current directory 
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle") 
print("Directory changed") 
print("\n") 

# data1 = loadmat('f_FFT_Nor_106_2b1_Pl_mc_LittC2SE_2.mat')
# print(data1['f'])

# data_raw = data1['f']
# pdata = pd.DataFrame(data_raw)
# print(pdata)

#---------------------------------------
#input data form fourier transform

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle")

for i in entries:
    data1 = loadmat(i)
    data_raw = data1['f']
    pdata = pd.DataFrame(data_raw)
    all_pdata = pdata.append(pdata)