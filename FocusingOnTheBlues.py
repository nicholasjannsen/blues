# -*- coding: utf-8 -*
"""
---------------------
SOFTWARE DESCRIPTION:
---------------------

Written October 2018 -- Nicholas Jannsen
Typeset in Python 3

This python module 
"""

# Numpy:
import numpy as np

# Astropy:
from astropy.io import fits

# Matplotlib:
import matplotlib.pyplot as plt

# Others:
import math, sys, time, scipy, glob, pylab

# Own functions:
from BlueSONG import BlueSONG
import Plot_Tools as pt

###########################################################################################################
#                                         FUNCTION: FocusingOnTheBlues                                    #
###########################################################################################################
          

# Load data:
path = '/home/nicholas/Data/SONG/'
FF_files = 


for i in range(len(FF_files)):

    # Load data:
    with fits.open(str(FF_files[i]))[0]:
        LF_i  = hdu_i.FF

        # Only cut out 2. order:
        FF = FF_all[100:500, :]

# Import software:
blues = BlueSONG(path, 'bs1', FF.T, FF.T)

# 
trace = blues.trace_orders(FF, LF, plot=0)





###########################################################################################################
#                                            DEFINE CLASS                                                 #
###########################################################################################################

class FocusingOnTheBlues(object):

    # INITILIZE THE CLASSE: 
    def __init__(self, path, img_name, plot=0, save=0):
        
        # DEFINE GLOBAL VARIABLES (DGV)

        # Customized information:
        self.path     = path       # Directory path to data
        self.img_name = img_name   # Name of images
        self.plot     = plot       # Plot if 1
        self.save     = save       # Save if 1

        # Header information:
        self.img_files = np.sort(glob.glob('{}{}*'.format(self.path, self.img_name)))
        self.n = len(self.img_files)
        self.h, self.w = 2200, 2750

        # Test multiple conditions:
        self.time = np.zeros(self.n)
        self.date = ['' for x in range((self.n))]
        for i in range(len(self.img_files)): 
        #self.img  = np.zeros((self.n, self))
            # Check data structure:
            hdulist  = fits.open(str(self.img_files[i]))
            header0 = hdulist[0].header
            # Load only flats:
            if header0['NAXIS1']==self.w and \
               header0['NAXIS2']==self.h and \
               header0['EXPTIME']==600 and \
               header0['IMAGETYP']=='flat':
                # Use the following images:
                self.time[i] = header0['JD-DATE']
                self.date[i] = header0['DATE-OBS']
                #self.img[i]  = pyfits.getdata(str(self.img_files[i]))
                #print self.img
                #print hdulist
                #sys.exit()
            # Image type:
            #img_type =  

        #print self.date
        #sys.exit()
        
        
        # try: # For now these has to be done manually
        #     #--- blue spec song ---#
        #     self.target  = header0['OBJECT']
        #     self.w       = header0['NAXIS1']   # Height of image 
        #     self.h       = header0['NAXIS2']   # Width  of image
        #     self.filt    = header0['INSFILTE']
        #     self.exptime = header0['EXPTIME']
        #     self.time    = Time(header0['DATE'], scale='utc').jd # Scaling to utc time and Julian Date 2000
            
        # except NameError: print('ERROR: FITS HEADER INFORMATION DO NOT MATCH PROGRAM VALUES'); return
        
###########################################################################################################
#                                            MAIN FUNCTIONS                                               #
###########################################################################################################

    def focus_using_flats(self):
        """
        text
        """
        #-----------
        # CONSTANTS:
        #-----------
        xmin, xmax, xcut = 300, 400, 1400
        xran = 100
        xcen = [340, 360, 360, 360, 360, 360, 360]

        #------------------
        # LOAD FLAT IMAGES:
        #------------------
        
        files = np.sort(glob.glob('{}{}*'.format(self.path, self.img_name)))
        LF_i  = np.array([fits.getdata(str(fil)) for fil in files])
        n    = len(files)
        h, w = np.shape(LF_i[0])

        #---------------
        # CUT OUT LINES:
        #---------------

        lines = np.zeros((n, h))
        for i in range(n):
            img = LF_i[i]
            lines[i] = img[:, xcut]

        #----------------------------
        # PLOT CROSS-LINES OF ORDERS:
        #----------------------------
            
        # Plot result:
        labels = self.date
        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, n)])
        for i in range(n):
            plt.plot(range(h), lines[i])
            labels.append('{}'.format(labels[i]))
        # The legend:
        plt.legend(labels, ncol=4, loc='upper center', 
                   bbox_to_anchor=[0.48, 1.1], 
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=False)
        plt.show()
        
        #-----------------------------
        # FIND FWHM TO VALIDATE FOCUS:
        #-----------------------------

        X = range(h)
        x = X[xmin:xmax]
        dex = (X>xmin)*(X<xmax)

        # Perform fit:
        fit = np.zeros((n,xran))
        for i in range(n):
            Yi = lines[i]
            y  = Yi[xmin:xmax]-np.median(Yi[xmin:xmax])
            from lmfit.models import GaussianModel
            model  = GaussianModel()
            result = model.fit(y, x=x, amplitude=max(y), center=xcen[i], width=40)
            fit[i] = result.best_fit
            print(labels[i])
            print(result.fit_report())

        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, n)])   
        for i in range(n):
            y = lines[i][xmin:xmax] - np.median(lines[i][xmin:xmax])
            plt.plot(x, y, '.-')
            #plt.plot(x, result.init_fit, 'k--')
            #plt.plot(x, fit[i], 'k-')
            labels.append('{}'.format(labels[i]))
        # The legend: 
        plt.legend(labels, ncol=4, loc='upper center', 
                   bbox_to_anchor=[0.48, 1.1], 
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=False)
        # Plot fits without labels:
        for i in range(n):
            plt.plot(x, fit[i], 'k-')
        plt.show()

    
    
