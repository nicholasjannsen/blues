# -*- coding: utf-8 -*
"""
---------------------
SOFTWARE DESCRIPTION:
---------------------

Written October 2018 -- Nicholas Jannsen
Typeset in Python 3

This python module is specifically made for the spectroscopic data reduction of the Shelyak eShel spectrograph which is installed at the Hertzsprung SONG node telescope at Tenrife, Spain. The software is originally built from structures of the 'SONGWriter' which is SONG's spectroscopic data reduction pipeline, and by others is inspired by the data reduction pipeline 'FIESTools' of the FIES spectrograph at the NOT on La Palma.
"""

# Numpy:
import numpy as np
# Astropy:
from astropy.io import fits
# Matplotlib:
import matplotlib.pyplot as plt
# Others:
import math, sys, time, scipy, glob, pylab
# Project functions:
from BlueSONG import BlueSONG
import Plot_Tools as pt

###########################################################################################################
#                                            FUNCTION: Blues                                              #
###########################################################################################################

#---------------------
# INITILIZE CLASS    :
#---------------------

# Masterfile:
#------------ Good:
path = '/home/nicholas/data/eshel/'  # sig^2 Eri (-42.18 km/s)
#path = '/home/nicholas/Data/song/obs/hd136202/' # 5 Ser   (+54.30 km/s)
#path = '/home/nicholas/Data/song/obs/hd142373/' # chi Her (-56.88 km/s)
#path = '/home/nicholas/Data/song/obs/hd152391/' #45.09     
#path = '/home/nicholas/Data/song/obs/hd185395/20171003/' # -27.26   # theta cyg
#------------ Poor:
#path = '/home/nicholas/Data/song/obs/hd185395/20171205/' # -27.26   # theta cyg

# Load software:
blues = BlueSONG(path, 'sun')

#---------------------
# CALIBRATION        :
#---------------------

print('1/10  - Image reduction')
S_calib, F_calib, T_calib = blues.image_reduction(redo=0, plot=1)

#---------------------
# ORDER TRACING      :
#---------------------

print('2/10  - Trace orders')
blues.trace_orders(plot=1)
sys.exit()
#---------------------
# INTER-ORDER MASK   :
#---------------------

print('3/10  - Inter-order mask')
blues.inter_order_mask(plot=0)

#---------------------
# FIND BACKGROUND    :
#---------------------

print('4/10  - Background subtraction')
F, F_back = blues.background(F_calib, poly_order_y=3, poly_order_x=7, plot=0)
S, S_back = blues.background(S_calib, poly_order_y=2, poly_order_x=4, plot=0)
# Extra plot when stellar background have been determined:
#pt.plot_background_residuals(F_back, S_back, S)

#---------------------
# SPECTRAL EXTRACTION:
#---------------------

print('5/10  - Spectral extraction')
blues.spectral_extraction(S, F, T_calib, plot=0)

#---------------------
# WAVELENGTH CALIB   :
#---------------------

print('6/10  - Wavelength calibration')
blues.wavelength_calib(plot=0)

#---------------------
# BLAZE FUNCTION     :
#---------------------

print('7/10  - De-blazing')
blues.deblazing(plot=0)

#---------------------
# MERGE ORDERS       :
#---------------------

print('8/10  - Scrunch and Merge orders')
blues.scrunch_and_merge(plot=0)

#---------------------
# RV CALIB           :
#---------------------

print('9/10  - RV calibration')
blues.rv_correction(plot=0)

#---------------------
# CONTINUUM NORMALIZE:
#---------------------

print('10/10 - Continuum normalization')
blues.continuum_norm(plot=0)

#---------------------
# ACTIVITY PROXIES   :
#---------------------

print('11/10 - Activity proxies')
blues.eshel_sindex(S, F, plot=1)


