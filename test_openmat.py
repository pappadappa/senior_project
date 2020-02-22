# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:43:31 2020

@author: Admin
"""
# import tables

# file = tables.openFile('test.mat')
# lon = file.root.lon[:]
# lat = file.root.lat[:]
# # Alternate syntax if the variable name is in a string
# varname = 'lon'
# lon = file.getNode('/' + varname)[:]

from scipy.io import loadmat
x = loadmat('f_FFT_Nor_106_2b1_Pl_mc_LittC2SE_1')
lon = x['lon']
lat = x['lat']
# one-liner to read a single variable
lon = loadmat('test.mat')['lon']