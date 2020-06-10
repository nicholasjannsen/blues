# -*- coding: utf-8 -*
"""
---------------------
SOFTWARE DESCRIPTION:
---------------------

Written October 2018 -- Nicholas Jannsen
Typeset in Python 3

This python class is specifically made for the spectroscopic data reduction of the Shelyak eShel spectrograph
which is installed at the Hertzsprung SONG node telescope at Tenrife, Spain. The software is originally built
from structures of the 'SONGWriter' which is SONG's spectroscopic data reduction pipeline, and by others is 
inspired by the data reduction pipeline 'FIESTools' of the FIES spectrograph at the NOT on La Palma.
"""

# Numpy:
import numpy as np
# Astropy:
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
# PyAstronomy:
import PyAstronomy as pyas
# SciPy:
import scipy
import scipy.constants
import scipy.io
import scipy.ndimage
from scipy.ndimage import median_filter
# Matplotlib:
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tikzplotlib import save as tikz_save
# Others:
import math, sys, time, glob, pylab, heapq
import bottleneck
from skimage import feature as skfeature
# Error of propagation (nominal_value, std_dev):
import uncertainties.unumpy as up
from uncertainties import ufloat
def val(x): return up.nominal_values(x)
def err(x): return up.std_devs(x)
# Own functions:
import Plot_Tools as pt

# Global settings for out-print to terminal (allow more digits and nice coloum ordering):
np.set_printoptions(suppress=True, formatter={'float_kind':'{:7.5f}'.format}, linewidth=100)

############################################################################################################
#                                               DEFINE CLASS                                               #
############################################################################################################

class BlueSONG(object):

    # INITILIZE THE CLASS: 
    def __init__(self, path, img_name):

        #-------------------------------
        # DEFINE GLOBAL VARIABLES (DGV):
        #-------------------------------
        
        # USER DEFINED VARIABLES:
        self.img_name   = img_name    # Name of image files
        self.path       = path        # Directory path to data
        self.path_img   = '/home/nicholas/Dropbox/thesis/latex/pictures/'
        self.path_blues = '/home/nicholas/Dropbox/Software/Python/blues/'
        self.cross_cut  = [50, 500]                     # Cut of spectral region in cross dispersion
        self.orders     = [1, 2]
        self.n_orders   = len(self.orders)
        
        # File handling:
        self.img_files = np.sort(glob.glob('{}{}*'.format(self.path, self.img_name)))
        self.hdul      = np.array([fits.open(str(files)) for files in self.img_files])
        # Extract headers and sepearte files:
        self.img_type = [self.hdul[i][0].header['IMAGETYP'] for i in range(len(self.img_files))]
        self.BF_dex = np.where(np.array(self.img_type)=='bias')[0]
        self.DF_dex = np.where(np.array(self.img_type)=='dark')[0]
        self.FF_dex = np.where(np.array(self.img_type)=='flat')[0]
        self.TA_dex = np.where(np.array(self.img_type)=='thar')[0]
        self.SF_dex = np.where(np.array(self.img_type)=='star')[0]
        # Open header of object:
        header = self.hdul[self.SF_dex[0]][0].header
        
        # Observation information:
        self.datetime  = header['DATE-OBS']         # Date and time of observation (string)
        self.date      = self.datetime[:10]         # Date of observation (string)
        self.jdate     = header['JD-DATE']          # Julian date (float)
        self.altitude  = header['OBJ-ALT']          # [deg] Initial altitude of target during obs (float) 
        self.seeing    = header['SEEING2']          # [arcsec] Running mean seeing on slit guiders (float)
        
        # Target information:
        self.target    = header['OBJECT']           # Name of target (string)
        self.ra        = header['OBJ-RA']           # Object Right Accension (string)
        self.dec       = header['OBJ-DEC']          # Object Declination (string)
        self.magnitude = header['OBJ-MAG']          # Magnitude of object (float)

        # Dimension constants and arrays:
        self.len_disp  = header['NAXIS1']           # [pixel] Height of image (int)
        self.len_cross = header['NAXIS2']           # [pixel] Width  of image (int)
        self.cen_disp  = int(self.len_disp/2)       # [pixel] Center position of disp  (int)
        self.cen_cross = int(self.len_cross/2)      # [pixel] Center position of cross (int)
        self.disp      = np.arange(self.len_disp)   # [pixel] Integers spanning disp (array)
                                   
        # eShel and CCD setup constants:
        self.res_power = 10000                      # Resolving power
        self.gain      = 0.27                       # [e-/ADU] Gain at -10 degC 
        self.ron       = 20                         # [e-] Read-out-noise
        self.pixel_size= header['PIXSIZE1']         # [micro m]

        # HK survey constants:
        self.V = 3901.000        # [Å] V quasi-continuum center 
        self.K = 3933.664        # [Å] Ca II K line
        self.H = 3968.470        # [Å] Ca II H line
        self.R = 4001.000        # [Å] R quasi-continuum center
        self.bands = [self.V, self.K, self.H, self.R]
        self.VR_bandpass = 20.0  # [Å] Bandpass of V and R continuums
        self.HK_bandpass = 1.09  # [Å] Bandpass of K and H lines
        
    ######################################################################################################
    #                                           CALIBRATION                                              #
    ###################################################################################################### 

    def image_reduction(self, redo=0, plot=0):
        """
        This routine takes data path and loads all image files given in the directory. 
        It combines the bias, dark, flat, and ThAr frames and make master frames used 
        for the image reduction of the science frames. The routine checks if master 
        calibrations frames already exists: (1) if they do it terminates, (2) if not it
        continues the image reduction. All calibration frames are saved with an extensions
        of the date, and science frames with the extension of the date and time of observation.
        ----------------------------
                    INPUT          :
        ----------------------------
        path               (string): Path to data
        plot              (integer): Plot flag activated by 1
        ----------------------------
                   OUTPUT          :
        ----------------------------
        BF_XX-XX-XX          (fits): Master bias
        DF_XX-XX-XX          (fits): Master dark 
        FF_XX-XX-XX          (fits): Master flat
        TA_XX-XX-XX          (fits): Master flat
        SF_XX-XX-XXTXX:XX:XX (fits): Science frame(s): Bias and dark frame calibrated light frames.
        """
        #------------------------------------------
        # TEST IF CALIBRATION IMAGES ALREADY EXIST:
        #------------------------------------------
        try:
            BF = fits.open('{}BF_{}.fits'.format(self.path, self.date))[0].data
            DF = fits.open('{}DF_{}.fits'.format(self.path, self.date))[0].data
            FF = fits.open('{}FF_{}.fits'.format(self.path, self.date))[0].data
            TA = fits.open('{}TA_{}.fits'.format(self.path, self.date))[0].data
            SF = fits.open('{}SF_{}.fits'.format(self.path, self.datetime))[0].data
        except IOError:
            BF = []

        #-------------------------
        # ELSE USE AVAILABLE DATA:
        #-------------------------
        
        if BF==[] or redo==1: 
            
            # Find all calibration images:
            BF_i = np.array([fits.getdata(str(self.img_files[i])) for i in self.BF_dex])
            DF_i = np.array([fits.getdata(str(self.img_files[i])) for i in self.DF_dex])
            FF_i = np.array([fits.getdata(str(self.img_files[i])) for i in self.FF_dex])
            TA_i = np.array([fits.getdata(str(self.img_files[i])) for i in self.TA_dex])
            SF_i = np.array([fits.getdata(str(self.img_files[i])) for i in self.SF_dex])
            
            # Exposure times:
            DF_exptimes = [self.hdul[self.DF_dex[i]][0].header['EXPTIME'] for i in range(len(self.DF_dex))]
            FF_exptimes = [self.hdul[self.FF_dex[i]][0].header['EXPTIME'] for i in range(len(self.FF_dex))]
            TA_exptimes = [self.hdul[self.TA_dex[i]][0].header['EXPTIME'] for i in range(len(self.TA_dex))]
            SF_exptimes = [self.hdul[self.SF_dex[i]][0].header['EXPTIME'] for i in range(len(self.SF_dex))]

            # Test if exposure times are the same:
            if int(np.sum(np.diff(DF_exptimes))) is not 0:
                print('ERROR: Dark exposure times are different!'); sys.exit()
            if int(np.sum(np.diff(FF_exptimes))) is not 0:
                print('ERROR: Flat exposure times are different!'); sys.exit()
            if int(np.sum(np.diff(TA_exptimes))) is not 0:
                print('ERROR: ThAr exposure times are different!'); sys.exit()
            
            #---------------------
            # PERFORM CALIBRATION:
            #---------------------
            
            # Make master bias:
            BF = np.median(BF_i, axis=0)
            # Make scaled master dark:
            DF_current = np.median(DF_i - BF, axis=0)
            DF_FF = (FF_exptimes[0]/DF_exptimes[0]) * DF_current
            DF_TA = (TA_exptimes[0]/DF_exptimes[0]) * DF_current
            DF    = (SF_exptimes[0]/DF_exptimes[0]) * DF_current
            # Make master flat:
            FF = np.median(FF_i - BF - DF_FF, axis=0)
            # Make master ThAr:
            TA = np.median(TA_i - BF - DF_TA, axis=0)
            # Calibrate science frames:
            SF = (SF_i - BF - DF)#/(FF/np.max(FF))

            #--------------------
            # SAVE MASTER FRAMES:
            #--------------------
            
            # Find hdulists:
            BF_hdul = self.hdul[self.BF_dex[0]][0].header
            DF_hdul = self.hdul[self.DF_dex[0]][0].header
            FF_hdul = self.hdul[self.FF_dex[0]][0].header
            TA_hdul = self.hdul[self.TA_dex[0]][0].header

            # Save master calibration images:
            fits.writeto('{}BF_{}.fits'.format(self.path, self.date), BF, BF_hdul, overwrite=True)
            fits.writeto('{}DF_{}.fits'.format(self.path, self.date), DF, DF_hdul, overwrite=True)
            fits.writeto('{}FF_{}.fits'.format(self.path, self.date), FF, FF_hdul, overwrite=True)
            fits.writeto('{}TA_{}.fits'.format(self.path, self.date), TA, TA_hdul, overwrite=True)

            # Save calibrated science frames one by one:        
            for i in range(len(self.SF_dex)):
                SF_hdul = self.hdul[self.SF_dex[i]][0].header
                header  = self.hdul[self.SF_dex[i]][0].header['DATE-OBS']
                fits.writeto('{}SF_{}.fits'.format(self.path, header), SF[0], SF_hdul, overwrite=True)
            # Only use first image if routine is running furter:
            SF = SF[0]

        #-----------------------------
        # LOAD RV AMPLITUDE OF OBJECT:
        #-----------------------------
        file_object = glob.glob('{}SF*'.format(self.path))
        hdul_object = fits.open(str(file_object[0]))
        self.rv_amp = hdul_object[0].header['OBJ-RV']   # [km/s] CDS RV amplitude (float)
        
        #-----------------------------------------------------------
        if plot==1: pt.plot_image_reduction(BF, DF, FF, TA, SF)
        #-----------------------------------------------------------
        # Select spectral region of interest:
        self.BF = BF; self.DF = DF; self.FF = FF; self.TA = TA
        self.F_calib = FF[self.cross_cut[0]:self.cross_cut[1], :].T
        self.T_calib = TA[self.cross_cut[0]:self.cross_cut[1], :].T
        self.S_calib = SF[self.cross_cut[0]:self.cross_cut[1], :].T
        self.noise   = np.sqrt(np.mean(BF**2))
        #-----------------------------------------------------------
        return self.S_calib, self.F_calib, self.T_calib
    
    ########################################################################################################
    #                                              FIND ORDERS                                             #
    ########################################################################################################

    def trace_orders(self, data=None, smooth_win=10, exclude_border=10, min_order_width=40, \
                     threshold_abs=0, disp_gap_tol=5, num_orders=5, num_peaks=10, plot=0):
        """
        This function find the orders in an eshel spectrum by tracing the maximum light distribution along
        each order. First the function finds a center order position and use this as a reference. Next the
        function finds the ridges of the specified number of order 'num_orders' using the skfeature package.
        Lastely, each order is the discribed by a 5 order polynomial and returned as output.
        ----------------------------
                    INPUT          :
        ----------------------------
        data                (array): A single image
        smooth_win          (float): Smooth value to enhance orders
        exclude_border      (float): Border edges that should be exluded
        order_min_width     (float): Minimum distance to locate where the orders are 
        threshold_abs       (float): Threshold used to locate peaks with skfeature
        disp_gap_tol        (float): Tolerance for how big a gap there may be
        num_orders          (float): User specified number of orders the program should find
        num_peaks           (float): Number of peaks found for each bin 
        ----------------------------
                   OUTPUT          :
        ----------------------------
        order_traces         (dict): Orders within 'order x' and corresponding array with polynomials
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if data==None: data = self.F_calib
        
        #----------------------------------
        # FIND CENTRAL REFERENCE POSITIONS:
        #----------------------------------
        
        # Central position interval 
        ref_int  = [self.cen_disp-5, self.cen_disp+6]
        
        ref_cen_pos = self.find_ref_cen_pos(data, ref_int, smooth_win, exclude_border, min_order_width,\
                                            threshold_abs, num_orders, plot)
        
        #------------------------
        # TRACE THE ORDER RIDGES:
        #------------------------
        
        ridge_pos_cross, ridge_pos_disp = self.find_order_ridges(data, smooth_win, exclude_border,\
                                                                 min_order_width, threshold_abs, num_peaks)
        #------------------------------------
        # FILL IN DATA INTO THE FOUND RIDGES:
        #------------------------------------
        
        # Make dict placeholders:
        order_traced = {}
        order_trace  = {}
        for i, order_pos in enumerate(np.sort(ref_cen_pos)[::-1]):
            # Here "order_pos" is the cross dispersion center value. order_pos[0] simply chooses one
            # value and not the increasing list within the loop.
            # Using ridges trace each order in each direction:
            min_order_width = 10
            order_trace_cross, order_trace_disp = self.find_order_outliers(self.cen_disp, order_pos[0],\
                                                                           ridge_pos_disp, ridge_pos_cross,\
                                                                           min_order_width, disp_gap_tol)
            # Fit ridges with polynomial:
            poly_coefs = np.polyfit(order_trace_disp, order_trace_cross, 5)
            order_traced['order_{}'.format(i)] = poly_coefs
            order_trace['order_{}'.format(i)]  = [order_trace_disp, order_trace_cross]

        #-----------------------------------------------------------------------------
        if plot==1:
            pt.plot_trace_order(ridge_pos_disp, ridge_pos_cross, order_trace, order_traced, \
                                order_trace_disp, self.cen_disp, ref_cen_pos)
        #-----------------------------------------------------------------------------
        self.ref_cen_pos = ref_cen_pos
        self.trace = order_traced
        #-----------------------------------------------------------------------------
        return order_traced
    

    def find_ref_cen_pos(self, data, ref_int, smooth_win, exclude_border, min_distance, threshold_abs, \
                         num_peaks, plot):
        """
        This function finds the center order position used as a reference.  
        """
        # Collapse in disp direction to reduce cosmic ray contamination:
        # (FIXME done to make this robust against cosmics - maybe it is not needed)
        center_rows_median = np.median(data[ref_int[0]:ref_int[1], :], axis=0)
        # Smooth cross_dispersion direction to prepare for the peak-detection algorithm:
        center_row_median_convolved = bottleneck.move_sum(center_rows_median.astype(np.float), \
                                                          smooth_win, min_count=1) 
        # Find orders using a peak detection function from scikit-image:
        order_centres = skfeature.peak_local_max(center_row_median_convolved, \
                                                 exclude_border=exclude_border,\
                                                 min_distance=min_distance, threshold_rel=0,\
                                                 threshold_abs=threshold_abs, num_peaks=num_peaks)
        # Peaks detected minus the smooth window applied (simply due to the moving sum of bottleneck):
        ref_cen_pos = order_centres - int(smooth_win/2)
        #------------------------------------------------------------------------------
        if plot==1:
            pt.plot_find_ref_cen_pos(center_rows_median, center_row_median_convolved, \
                                     self.len_cross, smooth_win, ref_cen_pos)
        #------------------------------------------------------------------------------
        return ref_cen_pos


    def find_order_ridges(self, data, smooth_win, exclude_border, min_distance, threshold_abs, num_peaks):
        """
        This function finds the ridge of each order. It does so by making a slice in cross dispersion and
        colvolve that with a smooth filter such as the "bottleneck.move_sum". It then finds the local max
        for each slice and saves the position
        """
        # Placeholders:
        ridge_indices_disp  = []
        ridge_indices_cross = []
        # Loop over the dispersion length (i) and the cross order row:
        for i, crossorder in enumerate(data):
            # Collapse in dispersion axis:
            # TODO should smoothing be handled separately?
            top_hat_conv = bottleneck.move_sum(crossorder.astype(np.float), smooth_win, min_count=1)
            # Again find the peaks as done in "find_ref_cen_pos":
            peaks = skfeature.peak_local_max(top_hat_conv, exclude_border=exclude_border,\
                                             min_distance=min_distance, threshold_rel=0,\
                                             threshold_abs=threshold_abs, indices=True, num_peaks=num_peaks)
            # Convert peaks to a list covering the ridges:
            peaks -= int(smooth_win/2)
            ridge_indices_cross = np.append(ridge_indices_cross, peaks)
            ridge_indices_disp  = np.append(ridge_indices_disp, np.ones(peaks.shape[0]) * i)
        #-----------------------------------------------------
        return ridge_indices_cross, ridge_indices_disp


    def find_order_outliers(self, cen_disp, ref_cen_cross, all_orders_x, all_orders_y, order_width,\
                              disp_gap_tol):
        """
        This utility takes the found reference positions in cross dispersion and the traced ridges and 
        locate all the outliers defined by 'order_width' threshold. If center_row is not an integer this
        will fail! 
        """
        # To simplify the code we make some abbreviations:
        x      = np.unique(all_orders_x)
        y_last = ref_cen_cross 
        x_last = x[cen_disp]
        cross_gap_tol = int(order_width/2.)
        # Placeholders for outliers:
        cross = []
        disp  = []
        # Outliers on the left side of cen_disp:
        for xi in x[cen_disp:]:
            index_xi = all_orders_x == xi
            orders_y = all_orders_y[index_xi]
            min_dist_index = np.argmin(np.abs(orders_y - y_last))
            new_y_pos = orders_y[min_dist_index]
            if (np.abs(new_y_pos - y_last) < cross_gap_tol) & (np.abs(xi - x_last) < disp_gap_tol):
                cross.append(new_y_pos)
                y_last = cross[-1]
                disp.append(xi)
                x_last = disp[-1]
        y_last = ref_cen_cross 
        x_last = x[cen_disp]
        # Outliers on the right side of cen_disp:
        for xi in x[cen_disp-1::-1]:
            index_xi = all_orders_x == xi
            orders_y = all_orders_y[index_xi]
            min_dist_index = np.argmin(np.abs(orders_y - y_last))
            new_y_pos = orders_y[min_dist_index]
            if (np.abs(new_y_pos - y_last) < cross_gap_tol) & (np.abs(xi - x_last) < disp_gap_tol):
                cross.append(new_y_pos)
                y_last = cross[-1]
                disp.append(xi)
                x_last = disp[-1]
        index = np.argsort(disp)
        #---------------------------------------------------
        return np.array(cross)[index], np.array(disp)[index]

    ########################################################################################################
    #                                           INTER-ORDER MASK                                           #
    ########################################################################################################

    def inter_order_mask(self, data=None, order_traces=None, order_width=None, \
                         low_nudge=0, high_nudge=0, plot=0):
        """
        This function is used to determine the background flux which will be used to correct for scattered
        light, wignetting, etc. The function looks at the flux level in between the orders ("inter-order") 
        and make and return a mask with ones for which is inter-order and zero elsewhere. The function uses
        the result from the previos subdroutine "traced orders".
        ----------------------------
                    INPUT          :
        ----------------------------
        order_width    (dict)      : Traced orders found from the function 'trace'
        order_traces   (int, float): Width of inter-order mask
        low_nudge      (int, float): Number of pixels used below the traced orders
        high_nudge     (int, float): Number of pixels used above the traced orders
        plot           (int, float): Plot result if you like
        ----------------------------
                   OUTPUT          :
        ----------------------------
        inter_order_mask (dict)    : Orders within 'order x' and corresponding array with polynomials
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if data        ==None: data         = self.F_calib
        if order_traces==None: order_traces = self.trace
        if order_width ==None: order_width  = self.find_optimal_width(plot=plot) # FUNCTION CALL!
        
        # Check if the inter-order width is odd integer:
        inter_order_width  = int(order_width * 4/3)
        if inter_order_width % 2 == 0: inter_order_width = inter_order_width - 1
               
        # Constants and placeholders:        
        inter_order_mask   = data * 0 + 1       # Initial image mask of ones 
        disp = np.arange(self.len_disp)         # Number pixel interval in dispersion
        order_no = sorted(order_traces.keys())  # Orders numbers (string)
        cross_order_center = []                    
        
        #-----------------------
        # FIND ALL INTER-ORDERS:
        #-----------------------
        
        # First loop through each order:
        for order in order_no:
            # Get the coefficients from the trace function:
            coefs = order_traces[order]                    
            cross_order_position = np.polyval(coefs, disp)  # Polyfit to each order
            cross_order_center = np.append(cross_order_center, cross_order_position[int(self.len_disp/2)])
            # Each inter order is found:
            for disp_i in range(self.len_disp):
                lower_order_edge =int(np.round(cross_order_position[disp_i]-inter_order_width/2-low_nudge))
                upper_order_edge =int(np.round(cross_order_position[disp_i]+inter_order_width/2+high_nudge))
                inter_order_mask[int(disp_i), lower_order_edge:upper_order_edge] = 0

        # Distance/size of each inter order:
        inter_order_size = cross_order_center[1:] - cross_order_center[:-1] - inter_order_width \
                           - low_nudge - high_nudge
        
        #-----------------------
        # REMOVE 'GHOST' ORDERS:
        #-----------------------
        
        # Predict inter_order_size:
        xx = np.arange(len(cross_order_center)-1)
        inter_order_size_fit = np.polyfit(xx, inter_order_size, 2)
        size_before = np.polyval(inter_order_size_fit, -1)
        size_after  = np.polyval(inter_order_size_fit, len(cross_order_center))
        
        # Remove 'ghost orders' before first order:
        coefs = order_traces[order_no[0]]
        cross_order_position = np.polyval(coefs, disp)
        for disp_i in range(self.len_disp):
            lower_inter_order_edge = np.round(cross_order_position[disp_i] - inter_order_width/2 \
                                              - low_nudge - size_before).astype(int)
            # Remove orders below edges:
            if lower_inter_order_edge < 0: lower_inter_order_edge = 0
            inter_order_mask[disp_i, :lower_inter_order_edge] = 0
            
        # Remove 'ghost orders' after last order:
        coefs = order_traces[order_no[-1]]
        cross_order_position = np.polyval(coefs, disp)
        for disp_i in range(self.len_disp):
            upper_inter_order_edge = np.round(cross_order_position[disp_i] + inter_order_width/2 \
                                              + high_nudge + size_after).astype(int)
            # Remove orders above edges:
            if upper_inter_order_edge > self.len_cross+50: upper_inter_order_edge = 0
            inter_order_mask[disp_i, upper_inter_order_edge:] = 0

        #--------------------------------------------------------------
        if plot==1: pt.plot_inter_order_mask(data, inter_order_mask)
        #--------------------------------------------------------------
        self.inter_order_width = inter_order_width
        self.inter_order_mask  = inter_order_mask
        #--------------------------------------------------------------
        return self.inter_order_mask

    ########################################################################################################
    #                                           BACKGROUND IMAGE                                           #
    ########################################################################################################

    def background(self, image, inter_order_mask=None, order_ref=None, \
                   poly_order_y=2, poly_order_x=4, filter_size=5, plot=0):
        """
        This function estimates the background flux of scattered light and subtract it. It uses the 
        inter_order_mask to perform this removal.   
        ----------------------------
                    INPUT          :
        ----------------------------
        mask             (2d array): Background mask with ones and zeros
        poly_order_x   (int, float): Order of polynomy to fits background flux in dispersion
        poly_order_y   (int, float): Order of polynomy to fits background flux in cross dispersion
        nsteps         (int, float): Number of steps  
        orderdef       (int, float):
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if inter_order_mask==None: inter_order_mask = self.inter_order_mask
        
        #----------------------------
        # CONSTANTS AND PLACEHOLDERS:
        #----------------------------
        
        # Create a background image:
        (ysize, xsize) = image.shape
        background_image = np.zeros((ysize, xsize),  dtype=np.float64)
        # Data size in arange: 
        xx = np.arange(xsize, dtype=np.float64)
        yy = np.arange(ysize, dtype=np.float64)
        # Array to withhold fitted y values:
        xfitarr = np.zeros((len(yy), xsize), dtype=np.float64)
        # Step size and range:
        yvals = np.arange(len(yy))
        # Constants:
        ycount       = 0
        niter        = 0
        sigma_limit  = 3
        # For plots:
        s_disp = [500, 1500, int(yvals[-1])]  # Slices in disp
        s_cros = [50,   200,    int(xx[-1])]  # Slices in cross
       
        #----------------------------
        # FIT IN Y-DIRECTION (CROSS):
        #----------------------------
        
        for i in yvals:
            # Cut out slice in cross dispersion with width determined by 'filter_size':
            ymin_ind = np.max([i - filter_size, 0])
            ymax_ind = np.min([i + filter_size, ysize-1])
            y_slice  = image[ymin_ind:ymax_ind, :]
            # Collapse in dispersion to a single cross row:
            y_mean = np.mean(y_slice, axis=0)
            # Indices/image of inter-order mask in cross row:
            y_image = np.where(inter_order_mask[i, :] == 1)[0]

            # Perform fitting with sigma-clipping: 
            while 1:
                # Make polynomial fit:
                coefs = np.polyfit(y_image, y_mean[y_image], poly_order_y)
                xfit  = np.polyval(coefs, y_image)
                # Find sigma:
                sigma     = (y_mean[y_image] - xfit) / np.std(y_mean[y_image] - xfit)
                rejected  = np.extract(sigma > sigma_limit, y_image)
                y_image   = np.extract(sigma < sigma_limit, y_image)
                # Loop until all image are within sigma or niter is reached:
                niter = niter + 1
                if niter == 5 or rejected.size == 0:
                    break

            # Final polynomial fit:
            xfit = np.polyval(coefs, xx)   # fitted line
            xfitarr[ycount, :] = xfit      # Array with fit constants for each slice
            ycount = ycount + 1

            # Save values for plotting:
            if i==s_disp[0]: yi0, ym0, yfit0 = y_image, y_mean[y_image], xfit
            if i==s_disp[1]: yi1, ym1, yfit1 = y_image, y_mean[y_image], xfit
            if i==s_disp[2]: yi2, ym2, yfit2 = y_image, y_mean[y_image], xfit
            
        #---------------------------
        # FIT IN X-DIRECTION (DISP):
        #---------------------------
        
        goodind  = np.arange(len(yy))
        for i in np.arange(xsize):

            # Perform fitting with sigma-clipping: 
            while 1:
                # Make polynomial fit:
                coefs = np.polyfit(yvals.take(goodind), xfitarr[goodind, i], poly_order_x)
                yfit  = np.polyval(coefs, yvals[goodind])
                # Find sigma: 
                sigma    = (xfitarr[goodind, i] - yfit) / np.std(xfitarr[goodind, i] - yfit)
                rejected = np.extract(sigma > sigma_limit, goodind)
                goodind  = np.extract(sigma < sigma_limit, goodind)
                # Loop until all image are within sigma:
                niter = niter + 1
                if niter == 3 or rejected.size == 0 or goodind.size == 0:
                    break

            # In case the image quality is higher than sigma_limit (poor quality image):
            if goodind.size == 0:
                print("Error: no points left when y-fitting the background")
                coefs = np.polyfit(xfitarr[:, i])

            # Final background image is constructed:
            background_image[:, i] = np.polyval(coefs, yy)

            # Save values for plotting:
            if i==s_cros[0]: xi0, xm0, xfit0 = yvals[goodind], xfitarr[goodind, i], background_image[:,i]
            if i==s_cros[1]: xi1, xm1, xfit1 = yvals[goodind], xfitarr[goodind, i], background_image[:,i]
            if i==s_cros[2]: xi2, xm2, xfit2 = yvals[goodind], xfitarr[goodind, i], background_image[:,i]

        #---------------------
        # SUBTRACT BACKGROUND:
        #---------------------
        corrected_image = image - background_image
        
        #--------------------------------------------------------------
        if plot is 1:
            pt.plot_background_fits(s_disp, s_cros, poly_order_y, poly_order_x, \
                                    xx, yi0, yi1, yi2, ym0, ym1, ym2, yfit0, yfit1, yfit2, \
                                    yy, xi0, xi1, xi2, xm0, xm1, xm2, xfit0, xfit1, xfit2)
            pt.plot_background(background_image)
        #--------------------------------------------------------------
        return corrected_image, background_image
    
    ########################################################################################################
    #                                           EXTRACT SPECTRUM                                           #
    ########################################################################################################
    
    def spectral_extraction(self, S, F, T, trace=None, order_width=None, plot=0):
        """
        This function uses the 'order_width' estimated earlier and first cut the spectrum in question using
        the utility 'cut_out_order'. All order of relevance is cut out, and a simple-sum over the spatial
        profile is used to get the 1D spectrum. Next clear cosmic hits are removed using the utility
        'locate_outliers' and finally the normalized flat blaze from each order is used to de-blaze each
        spectral order. 
        ----------------------------
                    INPUT          :
        ----------------------------
        S                (2d array): Stellar spectrum
        F                (2d array): Flat spectrum
        T                (2d array): ThAr arc spectrum
        trace                 (bib): subfunction with poly-fits to all the orders
        order_width           (int): Spatial order width for cutting out the order
        ----------------------------
                   OUTPUT          :
        ----------------------------
        s_deblaze        (1d array): De-blazed 1D stellar spectral orders
        T_orders         (2d array): ThAr arc image orders
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if trace      ==None: trace       = self.trace
        if order_width==None: order_width = self.order_width
        
        #----------------------------------------
        # FIRST ITERATION WITH LINEAR EXTRACTION:
        #----------------------------------------
        
        # Make sure that the order width is a odd number:
        if order_width % 2 == 0: order_width = order_width - 1

        # Cut out orders with spatial size (order number is coundt bottom-up):
        S_orders = [self.cut_out_order(S, np.polyval(trace['order_{}'.format(self.orders[i])], self.disp), \
                                       order_width) for i in range(self.n_orders)]
        F_orders = [self.cut_out_order(F, np.polyval(trace['order_{}'.format(self.orders[i])], self.disp), \
                                       order_width) for i in range(self.n_orders)]
        T_orders = [self.cut_out_order(T, np.polyval(trace['order_{}'.format(self.orders[i])], self.disp), \
                                       order_width) for i in range(self.n_orders)]

        # Linear extraction object and blaze:
        s_orders = [S_orders[i].sum(axis=1) for i in range(self.n_orders)]
        f_orders = [F_orders[i].sum(axis=1) for i in range(self.n_orders)]

        #------------------------------
        # HEREAFTER OPTIMAL EXTRACTION:
        #------------------------------
        # Initial variance image:
        #V = V0 + np.abs(S_calib, axis=0)/Q
        # # Cut out orders with spatial size 5*FWHM:
        # S = self.cut_out_order(np.polyval(trace['order_2'], self.disp), S_calib)
        # # Find extracted spectrum:
        # s = np.zeros(np.shape(s))
        # for i in range(self.len_disp):
        #     # Variance image:
        #     V = self.V0 + np.abs(s*P+S_sky, axis=0)/self.Q
        #     # Linear image:
        #     s[i] = np.sum((P*S_sky/V)/(P**2/V), axis=1)
        
        #--------------------------------------------------------------
        #if plot==1: pt.plot_optimal_extraction(S_orders[1][:, 900:950].T)
        #--------------------------------------------------------------
        self.S_orders = S_orders; self.F_orders = F_orders; self.T_orders = T_orders
        self.s_orders = s_orders; self.f_orders = f_orders
        #--------------------------------------------------------------
        return [self.S_orders, self.F_orders, self.T_orders]


    def cut_out_order(self, image, traced_order, cross_order_width=21):
        """
        This utility takes the polynomial describtion 'traced_order' and the relevant
        order and a spectrum, and cuts out the spectrum in total dispersion length and
        'cross_order_width' pixels in cross dispersion around this spectral order. It
        the returns the bandpass image 'order_cut' and positions which can be used for
        an easy way of flotting the result. 'cross_order_width' needs to be odd number. 
        """   
        # Conatant and placeholders:
        half_width            = int(cross_order_width/2.)
        order_cut             = np.zeros((self.len_disp, cross_order_width))
        cross_order_positions = np.zeros((self.len_disp, cross_order_width))

        # This loop cuts out the order:
        for d in np.arange(self.len_disp):
            position = traced_order[d]
            rounded_position = int(np.round(position))
            # Fill in the columns of the order:
            cp = image[d, rounded_position - half_width:rounded_position + half_width + 1]
            order_cut[d,:] = cp
            # Fill in the cross order position:
            x = np.arange(-half_width, half_width + 1) + rounded_position
            cross_order_positions[d, :] = x
        #--------------------------------------------------------------
        return order_cut

    ########################################################################################################
    #                                        WAVELENGTH CALIBRATION                                        #
    ########################################################################################################
    
    def wavelength_calib(self, T_orders=None, poly_order=3, plot=0):
        """ 
        This utility performs the wavelength calibration. This is done first using the Ca II lines as a 
        initial reference of the wavelength scale. Thus, the utility 'calcium_lines_identifier' finds the
        peaks values of the H and K lines, then 'peak_finder' finds all ThAr lines in the order above a
        certain threshold, and lastly a FIES ThAr atlas is used to set the real wavelength scale of the
        order, which is returned as output. The terminology here is that (p) is pixel wavelength, (l) is
        wavelenght in Å, and (x) is the spatial direction.
        ----------------------------
                    INPUT          :
        ----------------------------
        s_orders         (1d array): Stellar spectrum for each order
        T_orders         (2d array): Extracted ThAr image of each order 
        poly_order            (int): Order of polynomial function for fitting wavelength relation
        ----------------------------
                   OUTPUT          :
        ----------------------------
        [l0, l1]        (1d arrays): New wavelength scale for each order
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if T_orders==None: T_orders = self.T_orders

        #-------------------------
        # FIND OBSERVED ARC LINES:
        #-------------------------

        # Identify lines from different sigma levels:
        COF, radii = np.zeros(self.n_orders), np.zeros(self.n_orders)
        for i in range(self.n_orders):
            COF_i, radii_i, _ = self.peak_finder(T_orders[i], sigma=0.5, plot=0)
            COF     = [COF[i], COF_i]
            radii   = [radii[i], radii_i]

        # Only keep disp values:
        l_cof0 = COF[0][:,0]
        l_cof1 = COF[1][:,0]

        #---------------------------------------------------
        # ITER 1: USE KNOWN FIES LINES AS INITIAL REFERENCE: TODO! automate the first iteration
        #---------------------------------------------------

        # Cacium lines:
        l_ca = [self.K, self.H]

        # Load FIES arc atlas:
        l_fies0 = [3868.5284, 3873.8224, 3916.4176, 3925.7188, 3928.6233, 3948.9789]
        l_fies1 = [3925.7188, 3928.6233, 3948.9789, 3950.3951, 3979.3559, 4013.8566]

        # Pixel coordinats to known identified FIES peaks:
        l_pix0 = np.array([1091, 1190, 2025, 2217, 2279, 2723])
        l_pix1 = np.array([ 900,  952, 1326, 1352, 1910, 2624])

        # Find COF lines closest to known pixel coordinates:
        l_cof0_ini = [min(l_cof0, key=lambda x:abs(x-l_pix0[i])) for i in range(len(l_pix0))]
        l_cof1_ini = [min(l_cof1, key=lambda x:abs(x-l_pix1[i])) for i in range(len(l_pix1))]
        
        # Find wavelenght relations:  
        r0_1 = self.find_arc_scale(self.disp, l_cof0_ini, l_fies0, poly_order, l_cof0, \
                                   param=['1. ITER: 58', T_orders[0], COF[0], radii[0]], plot=plot)
        r1_1 = self.find_arc_scale(self.disp, l_cof1_ini, l_fies1, poly_order, l_cof1, \
                                   param=['1. ITER: 57', T_orders[1], COF[1], radii[1]], plot=plot)        
        l0_ini, l_cof0_ini, res0_ini, l_cof0_all = r0_1[0], r0_1[1], r0_1[2], r0_1[3]
        l1_ini, l_cof1_ini, res1_ini, l_cof1_all = r1_1[0], r1_1[1], r1_1[2], r1_1[3]
        
        #--------------------------------------
        # ITER 2: CALIBRATE WITH PHOTRON ATLAS:
        #--------------------------------------
        
        # Load Photron arc atlas (http://iraf.noao.edu/specatlas/thar_photron/thar_photron.html):
        thar_atlas_phot = np.loadtxt('dependencies/thar_atlas_photron.txt')
        l_atlas_phot = thar_atlas_phot[:,2]

        # Find wavelenght relations:
        r0_2 = self.find_arc_scale(l0_ini, l_cof0_all, l_atlas_phot, poly_order, threshold=0.02, \
                                   param=['2. ITER: 58', T_orders[0], COF[0], radii[0]], plot=plot)
        r1_2 = self.find_arc_scale(l1_ini, l_cof1_all, l_atlas_phot, poly_order, threshold=0.02, \
                                   param=['2. ITER: 57', T_orders[1], COF[1], radii[1]], plot=plot)
        l0, l_cof0, std0 = r0_2[0], r0_2[1], r0_2[2]
        l1, l_cof1, std1 = r1_2[0], r1_2[1], r1_2[2]

        #--------------------------------------------------------------
        if plot==1: pt.plot_arc_check([l0, l1], T_orders, l_ca, 'FINAL RESULT')
        #--------------------------------------------------------------
        self.l_orders = [l0, l1]
        self.sigma_w  = [std0, std1]
        #--------------------------------------------------------------
        return self.l_orders

    
    def find_arc_scale(self, l_scale, l_obs, l_atlas,  poly_order, l_all=None, \
                       threshold=None, param=None, plot=None):
        """
        This utility takes observed and atlas arc lines and find a wavelength solution given a threshold
        for comparing when the lines are close enough to be identified as a match. Thus, this utility works
        only if given a fair initial wavelength solution for the observed lines. If no threshold is given it 
        will be assumed that the this is the initial step going from pixel to wavelenght space, where the 
        exact coordinate matches are known.
        """
        if threshold is not None:
            # Find atlas lines closest to observed lines (COF lines):
            l_atlas_match = [min(l_atlas, key=lambda x:abs(x-l_obs[i])) for i in range(len(l_obs))]
            
            # Find value difference between matched lines and keep only if diff < 1 Å:
            dex_goodin = np.where(abs(l_obs - l_atlas_match) < threshold)[0]
            l_atlas_good = [l_atlas_match[dex_goodin[i]] for i in range(len(dex_goodin))]
            l_obs_good   = [l_obs[dex_goodin[i]]         for i in range(len(dex_goodin))]
        else:
            l_obs_good   = l_obs.copy()
            l_atlas_good = l_atlas.copy()
        # Find new pixel-wavelength relation:
        coefs = np.polyfit(l_obs_good, l_atlas_good, poly_order)
        
        # Copy solution to scale and observed lines:
        ipoly = np.arange(poly_order+1)[::-1]
        l_scale_new = np.sum([coefs[i]*l_scale**ipoly[i] for i in range(poly_order+1)], axis=0)
        l_obs_new   = np.sum([coefs[i]*l_obs**ipoly[i]   for i in range(poly_order+1)], axis=0)
        if l_all is not None:
            l_all_new = np.sum([coefs[i]*l_all**ipoly[i] for i in range(poly_order+1)], axis=0)
        else: l_all_new = None

        # Calculate fit parameters:
        poly      = np.poly1d(coefs)
        xp        = np.linspace(min(l_obs_good), max(l_obs_good), 1e3)
        residuals = poly(l_obs_good)-l_atlas_good
        chi2r     = 1-residuals/(len(l_obs_good)-poly_order+1)
        sigma     = np.std(residuals)

        #--------------------------------------------------------------
        if plot is 1:
            # Activate only illustration and save:
            #pt.plot_arc_illustration(l_obs, l_atlas, l_obs_good, l_atlas_good, l_scale, param)
            pt.plot_arc_fit(l_obs_good, l_atlas_good, coefs, poly_order, residuals, chi2r, sigma, param[0])
            pt.plot_arc_scale(l_obs, l_atlas, l_obs_good, l_atlas_good, l_scale, param)
        #--------------------------------------------------------------
        return l_scale_new, l_obs_new, sigma, l_all_new

    ########################################################################################################
    #                                               DE-BLAZING                                             #
    ########################################################################################################
    
    def deblazing(self, F_orders=None, l_orders=None, f_orders=None, s_orders=None, plot=0):
        """
        This spectral order. 
        ----------------------------
                    INPUT          :
        ----------------------------
        T                (2d array): ThAr arc spectrum
        trace                 (bib): subfunction with poly-fits to all the orders
        order_width           (int): Spatial order width for cutting out the order
        ----------------------------
                   OUTPUT          :
        ----------------------------
        s_deblaze        (1d array): De-blazed 1D stellar spectral orders
        T_orders         (2d array): ThAr arc image orders
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if F_orders==None: F_orders = self.F_orders
        if l_orders==None: l_orders = self.l_orders
        if f_orders==None: f_orders = self.f_orders
        if s_orders==None: s_orders = self.s_orders
        
        #----------------------
        # CORRECT FOR COMSMICS: TODO! IF using optimal extraction this can be fixed at once
        #----------------------
        # Do not work for arc images ask it is a peak detection algorithm:
        f_lincor = [self.locate_outliers(f_orders[i], convolve_step=3, cutoff=2e-2, plot=0) \
                    for i in range(self.n_orders)]
    
        #----------------------------
        # CORRECT FOR BLAZE FUNCTION:
        #----------------------------
        
        # Perform blaze correction:
        s_deblaz = [(s_orders[i]/f_lincor[i]) for i in range(self.n_orders)]

        #-----------------------------------
        # SCALE FLAT BALZE TO STELLAR BALZE:
        #-----------------------------------

        # Find maximum of each blaze:
        dex_blaze_max = [np.nanargmax(self.f_orders[i]) for i in range(self.n_orders)]

        # Remove all cosmics from spectra to be used only for scaling:
        s_coscor = [self.locate_outliers(s_orders[i], convolve_step=3, cutoff=1e-1, plot=0) \
                    for i in range(self.n_orders)]

        # With cosmics removed now scale to maximum difference:
        continuum_cor = 0.85 # Correction factor only valid for solar type stars
        dif_max = [continuum_cor*np.max(self.f_orders[i])/np.max(s_coscor[i]) for i in range(self.n_orders)] 
           
        #--------------------------------------------------------------
        if plot==1:
            pt.plot_blaze(s_orders, f_orders, f_lincor, dif_max)
            pt.plot_deblaze(s_deblaz)
        #--------------------------------------------------------------
        self.f_orders = f_lincor; self.s_deblaz = s_deblaz
        self.dif_max = dif_max
        #--------------------------------------------------------------
        return self.f_orders, self.s_deblaz

    
    def norm_blaze_function(self, F_order): #TODO! this is not used but may be useful in future
        """
        This utility finds the blaze function which recides from the fact that an échelle spectrum is
        bowed along the dispersion and thus... To find the blaze function here the order is collapsed
        in cross dispersion to a one dimentional array using a simple sum. The 'normalized_order' also
        gives an estimate of scatter within the order.
        """
        # Use simple sum to collapse order:
        f_blaze = np.sum(F_order, axis=1)
        # Normalize the spectrum:
        F_norm_order = np.zeros(F_order.shape)
        for i in range(F_order.shape[1]):
            F_norm_order[:, i] = F_order[:, i] / f_blaze
        #--------------------------------------------------------------
        return F_norm_order

    #F_norm = [self.norm_blaze_function(F_orders[i]) for i in range(self.n_orders)]
    #S_deblaze = [self.S_orders[i]/F_norm[i] for i in range(self.n_orders)]
    #s_deblaz_norm = [S_deblaze[i].sum(axis=1) for i in range(self.n_orders)]

    #####################################################################################################
    #                                  SCRUNCH, MERGE, AND CLIP                                         #
    #####################################################################################################

    def scrunch_and_merge(self, l_orders=None, s_deblaz=None, plot=0):
        """
        This function estimates    
        ----------------------------
                    INPUT          :
        ----------------------------
        mask             (2d array): Background mask with ones and zeros
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if l_orders==None: l_orders = self.l_orders
        if s_deblaz==None: s_deblaz = self.s_deblaz
        
        #----------------
        # SCRUNCH ORDERS:
        #----------------

        # Prepare uniform wavelength grid:
        dl        = np.median([np.median(np.diff(l_orders[i])) for i in range(self.n_orders)])
        dl_orders = [np.arange(l_orders[i][0], l_orders[i][-1]+dl, dl) for i in range(self.n_orders)]

        # Linear interpolate to uniform grid:
        s_grids = [scipy.interpolate.griddata(l_orders[i], s_deblaz[i], dl_orders[i], method='nearest') \
                   for i in range(self.n_orders)]
        
        #--------------
        # MERGE ORDERS:
        #--------------
        
        # Find index of merge boarders:
        dex0_min = np.where(dl_orders[0].astype(int)==3880)[0][0]
        dex0_max = np.where(dl_orders[0].astype(int)==3915)[0][0]
        dex1_min = np.where(dl_orders[1].astype(int)==3915)[0][0]

        # Merge wavelength axis:
        l_merge0 = dl_orders[0][dex0_min:dex0_max]
        l_merge1 = dl_orders[1][dex1_min::]
        l_merge    = np.append(l_merge0, l_merge1)

        # Merge flux axis:
        s_merge0 = s_grids[0][dex0_min:dex0_max]
        s_merge1 = s_grids[1][dex1_min::]
        s_merge  = np.append(s_merge0, s_merge1)

        #--------------------------------------------------------------
        if plot is 1: pt.plot_merge(s_merge, l_merge, [self.H, self.K])
        #--------------------------------------------------------------
        self.s = s_merge; self.l = l_merge
        #--------------------------------------------------------------
        return self.s, self.l

    #####################################################################################################
    #                                         RV CORRECTION                                             #
    #####################################################################################################

    def rv_correction(self, s=None, l=None, plot=0):
        """
        This function is used to create a transit model for the Cross-Correlation (CC). 
        To create the model the subrutine called 'model' is used. To perform the CC the
        CC coefficient is also needed and this is calculated in the subroutine 
        'cc_coefficients'. For the precision needed here future test of RV correction is
        needed. For now Astropy's find the projected RV component including the stellar
        motion, the baryocentric motion, and Earth's heliocentric and rotational velocities.
        ----------------------------
                    INPUT          :
        ----------------------------
        mask             (2d array): 
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """
        # Check if program parameters is defined:
        if s==None: s = self.s
        if l==None: l = self.l

        #---------------
        # RV CORRECTION:
        #---------------

        # Use astropy for Barycentric:
        obstime  = Time(self.datetime)
        target   = SkyCoord.from_name(self.target)  
        song     = EarthLocation.of_site('roque de los muchachos') # Closest observatory to Teide 
        rv_baryc = target.radial_velocity_correction(obstime=obstime, location=song).to('km/s').value

        # Use baryocentric + stellar RV amplitude as RV correction:
        rv_shift = self.rv_amp - rv_baryc - self.rv_amp * rv_baryc / (scipy.constants.c/1e3)
        
        # Use standard equation for RV shift (dl/l = v/c) calculated in [km/s]:
        c = scipy.constants.c/1e3
        delta_lambda = rv_shift / c * l
        
        # Perform wavelenght shift:
        l = l - delta_lambda

        # Calculate this for approx results:
        delta_l3950 = rv_shift / (scipy.constants.c/1e3) * 3950
        delta_p3950 = delta_l3950 / np.diff(l)[0]
        
        #---------------------------
        # HANDLING IDL SUN SPECTRUM:
        #---------------------------

        # # Save IDL format to python:
        # import scipy.io
        # s_sun = scipy.io.readsav('{}sun_reference/ARCTURUS.IDL'.format(self.path_blues))
        # np.savetxt('{}sun_reference/sun_python.txt'.format(self.path_blues), s_sun['atlas'])

        # # Import sun spectrum and save smaller spectral domain:
        # sun   = np.loadtxt('{}/sun_reference/sun_python.txt'.format(self.path_blues))
        # l_sun = np.array([sun[i][0] for i in range(len(sun))])
        # s_sun = np.array([sun[i][2] for i in range(len(sun))])

        # plt.figure()
        # plt.plot(l_sun, s_sun, 'k-', linewidth=0.1)
        # plt.show()

        # # Find index of merge boarders:
        # borders = [3880, 4020]
        # i_min = np.where(l_sun.astype(int)==borders[0])[0][0]
        # i_max = np.where(l_sun.astype(int)==borders[1])[0][0]

        # # Merge wavelength axis and save data:
        # l_sun = l_sun[i_min:i_max]
        # s_sun = s_sun[i_min:i_max]
        # np.savetxt('{}sun_reference/sun.txt'.format(self.path_blues), np.vstack([l_sun, s_sun]).T)

        #-------------------------
        # TRANSFORM AND NORMALIZE:
        #-------------------------

        # # Load sun data: 
        # sun   = np.loadtxt('{}/sun_reference/sun.txt'.format(self.path_blues))
        # l_sun = sun[:,0]
        # s_sun = sun[:,1]
        
        # # Tranform to regular grid:
        # dl_sun = np.diff(l_sun)[0]
        # l_gsun = np.arange(l_sun[0], l_sun[-1], dl_sun)
        # # Interpolate:
        # s_gsun = scipy.interpolate.griddata(l_sun, s_sun, l_gsun, method='cubic')   # Sun
        # s_gobs = scipy.interpolate.griddata(l,     s,     l_gsun, method='nearest') # Star observed
        
        # # Flux normalize signals:
        # s_gobs = s_gobs/np.max(s_gobs)

        # # Inverted and Normalized signal MUST be used:
        # x = (1-s_gobs) - np.mean(1-s_gobs)
        # y = (1-s_gsun) - np.mean(1-s_gsun)

        # #x = self.convolve(x, 'median', 1000)
        # #y = self.convolve(y, 'median', 1000)
        # print(np.diff(l_gsun))
        # print(len(l_gsun))
        # l_gsun_new = np.roll(l_gsun, 200)

        # plt.figure()
        # plt.plot(l_gsun, s_gsun, 'r-')
        # plt.plot(l_gsun_new, s_gsun, 'b-')
        # plt.plot(l_gsun, s_gobs, 'k-')
        # plt.show()
        #sys.exit()
        
        #---------------------------
        # PERFORM CROSS CORRELATION: 
        #---------------------------

        # # Prepare indices for spectrum shift:
        # dx = 200
        # dy = np.arange(-dx, dx)
        
        # # Perform Cross-correlation:
        # cc = np.zeros(len(dy))
        # for i in dy:
        #     y = np.roll(y, -i)
        #     r_cc  = self.cc_coefficient(x, y)
        #     cc[i] = r_cc

        # # Find peaks maximum:
        # peaks_dex, _   = scipy.signal.find_peaks(cc)
        # dy_peaks = dy[peaks_dex]
        # cc_peaks = cc[peaks_dex]

        # # Choose only resonable RV shifts (<200 km/s) and good cc (>0.8):
        # peaks_good_dex = (dy_peaks>-200) * (dy_peaks<200) * (cc_peaks>0.8)
        # dy_peaks = dy_peaks[peaks_good_dex]
        # cc_peaks = cc_peaks[peaks_good_dex]

        # Find max peak corresponding to RV:
        # cc_max_dex = np.argmax(cc_peaks)
        # dy_max = dy_peaks[cc_max_dex]
        # cc_max = cc_peaks[cc_max_dex]
       
        # plt.figure()
        # plt.plot(dy, cc, 'k-')
        # plt.plot(dy_peaks, cc_peaks, 'r+')
        #plt.title('RV shift = {} km/s'.format(rv_mean))
        #plt.axvline(max(cc), color='r')
        #plt.plot(l_sun, y, 'k-')
        #plt.plot(ll, x, 'b-')
        #plt.xlim(-200, 200)
        # plt.show()
     
        #--------------------------------------------------------------
        #if plot is 1: pt.plot_merge(s_merge, l_merge, [self.H, self.K])
        #--------------------------------------------------------------
        self.delta_v_baryc = rv_baryc
        self.delta_v = rv_shift
        self.delta_l = delta_l3950
        self.delta_p = delta_p3950
        self.l = l
        #--------------------------------------------------------------
        return self.l

    
    def cc_coefficient(self, x, y): 
        """
        This function find the cross-correlation coefficienten between two datasets. Here x is the data
        have an offset and y is the data that is cross correlated for every small step.
        """
        cor  = np.sum( (x-np.mean(x)) * (y-np.mean(y)) )
        norm = np.sqrt( np.sum((x-np.mean(x))**2) * np.sum((x-np.mean(x))**2) )
        r    = cor/norm
        #--------------------------------------------------------------
        return r
    
    ########################################################################################################
    #                                      CONTINUUM NORMALIZATION                                         #
    ########################################################################################################

    def continuum_norm(self, l=None, s=None, rv_amp=0, plot=0):
        """
        This function estimates    
        ----------------------------
                    INPUT          :
        ----------------------------
        mask             (2d array): Background mask with ones and zeros - uses rv corrected data only!
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """
        #------------------------------
        # CHECK FOR PROGRAM PARAMETERS:
        #------------------------------
        if l==None: l = self.l
        if s==None: s = self.s

        #----------------------------
        # NORMALIZE WITH POINTS ONLY:
        #----------------------------

        # Pseudo-continuum points used for the SSS included in the Ca H & K order:
        ps_min, ps_max = 3912, 4000

        # Find central index for wavelength:
        dex_min = np.where(l.astype(int)==ps_min)[0][0]
        dex_max = np.where(l.astype(int)==ps_max)[0][0]

        # Values for points used to linear relation:
        l_point = [l[dex_min], l[dex_max]]
        s_point = [s[dex_max], s[dex_max]]

        # Find linear relation:
        coefs_point  = np.polyfit(l_point, s_point, 0)
        poly_point   = np.poly1d(coefs_point)
        s_norm_point = s/poly_point(l)
         
        #-----------------------------
        # NORMALIZE WITH HIGHEST PEAK: 
        #-----------------------------
        
        # Find max peak around pseudo peaks:
        # (This function needs a initial guess for the RV shift!)
        dex_peak_min = self.find_peak_in_noise(s, dex_min, plot=0)
        dex_peak_max = self.find_peak_in_noise(s, dex_max, plot=0)
        
        # Values for fit:
        l_peak = [l[dex_peak_min], l[dex_peak_max]]
        s_peak = [s[dex_peak_min], s[dex_peak_max]] 

        # Find linear relation:
        coefs_peak  = np.polyfit(l_peak, s_peak, 1)
        poly_peak   = np.poly1d(coefs_peak)
        s_norm_peak = s/poly_peak(l)
        
        #------------------------------
        # NORMALIZE WITH MEAN BANDPASS:
        #------------------------------
        
        # Another methods is to use pseudo-continuum bandpass':
        bold_band_min = (l > ps_min-0.5)*(l < ps_min+0.5)
        bold_band_max = (l > ps_max-0.5)*(l < ps_max+0.5)

        # Find meadian bandpass values:
        l_mean = [np.median(l[bold_band_min]), np.median(l[bold_band_max])]
        s_mean = [np.median(s[bold_band_min]), np.median(s[bold_band_max])]

        # Find linear relation:
        coefs_mean = np.polyfit(l_mean, s_mean, 1)
        poly_mean  = np.poly1d(coefs_mean)
        s_norm_mean = s/poly_mean(l)
        
        #--------------------------------------------------------------
        if plot is 1:
            pt.plot_continuum_norm_all(l, s, [l_point, l_peak, l_mean],[s_point, s_peak, s_mean], \
                                       [s_norm_point, s_norm_peak, s_norm_mean], \
                                       [poly_point, poly_peak, poly_mean], [self.K, self.H])
        #--------------------------------------------------------------
        self.l = l
        self.s = [s_norm_peak, s_norm_point, s_norm_mean]
        #--------------------------------------------------------------
        return self.l, self.s
   
    ########################################################################################################
    #                                               S-INDEX                                                #
    ########################################################################################################

    def eshel_sindex(self, S, F, trace=None, order_width=None, l_orders=None, f_orders=None, \
                     l=None, s=None, plot=0):
        """
        This function estimates    
        ----------------------------
                    INPUT          :
        ----------------------------
        mask             (2d array): Background mask with ones and zeros
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """
        #--------------------------
        # CHECK PROGRAM PARAMETERS:
        #--------------------------
        if trace==None: trace = self.trace
        if order_width==None: order_width = self.order_width
        if l_orders==None: l_orders = self.l_orders
        if f_orders==None: f_orders = self.f_orders
        if l ==None: l = self.l
        if s ==None: s = self.s
        
        #------------------
        # FIND UNCERTAINTY:
        #------------------
        self.uncertainty(S, F, trace, order_width, l_orders, f_orders, l, s[0], plot=1)

        #---------------------------------------------
        # FIND FLUX AND UNCERTAINTY FOR EACH BANDPASS:
        #--------------------------------------------
        result0 = self.find_bandpass_fluxes(l, s[0])
        result1 = self.find_bandpass_fluxes(l, s[1])
        result2 = self.find_bandpass_fluxes(l, s[2])
        results = [result0, result1, result2]

        #--------------
        # FIND S INDEX:
        #--------------
        sindices = [self.sindex(results[i], l, save=i) for i in range(len(results))]
        # Find fractional difference between each continuum method:
        sindex_diff12 = abs(sindices[0] - sindices[1]) / sindices[0] * 100 
        sindex_diff13 = abs(sindices[0] - sindices[2]) / sindices[0] * 100 
        print(sindex_diff12)
        print(sindex_diff13)  
        #--------------------------------------------------------------
        if plot is 1:
            self.sindex = sindices
            self.results()
        #--------------------------------------------------------------
        return

    
    def uncertainty(self, F, S, trace, order_width, l_orders, f_orders, l, s, plot=0):
        """
        This utility estimates the mean flux-uncertainty in a given bandpass using S/N ratio.
        """
        #-------------------------------------
        # S/N RATIO OF ALONG EACH ORDER BLAZE:
        #-------------------------------------
        # Calculate mean sky background from background subtracted image:
        self.f_flux_sky, _ = self.mean_background(F, trace, plot=0)
        self.s_flux_sky, _ = self.mean_background(S, trace, plot=0)
        # Scale flat orders to that of the stellar orders:
        f_flux_obj =  f_orders
        s_flux_obj = [f_flux_obj[i]/self.dif_max[i] for i in range(self.n_orders)]
        # Convolve data for smoothing:
        f_conv_obj = [self.convolve(f_flux_obj[i], 'mean', 10) for i in range(self.n_orders)]
        s_conv_obj = [self.convolve(s_flux_obj[i], 'mean', 10) for i in range(self.n_orders)]     
        # Find SNR along the orders:
        f_snr_orders = np.zeros(self.n_orders)
        s_snr_orders = np.zeros(self.n_orders)
        for i in range(self.n_orders):
            fsnr = [self.signal_to_noise(f_conv_obj[i][j], order_width, self.f_flux_sky) \
                   for j in range(self.len_disp)]
            ssnr = [self.signal_to_noise(s_conv_obj[i][j], order_width, self.s_flux_sky) \
                   for j in range(self.len_disp)]
            f_snr_orders = [f_snr_orders[i], fsnr]
            s_snr_orders = [s_snr_orders[i], ssnr]  
        # Find SNR peak maxima:
        self.f_snr_max = [max(f_snr_orders[i]) for i in range(self.n_orders)]
        self.s_snr_max = [max(s_snr_orders[i]) for i in range(self.n_orders)]

        #----------------------------
        # UNCERTAINTY FROM S/N RATIO:
        #----------------------------
        # Min and max indices for each bandpass:
        l_min = [self.V-self.VR_bandpass, self.K-self.HK_bandpass, \
                 self.H-self.HK_bandpass, self.R-self.VR_bandpass]
        l_max = [self.V+self.VR_bandpass, self.K+self.HK_bandpass, \
                 self.H+self.HK_bandpass, self.R+self.VR_bandpass]
        # Order nr to loop over:
        j = [0, 1, 1, 1]
        # Find mean S/N and uncertainty in each bandpass:
        self.f_snr_X = np.zeros(4)
        self.s_snr_X = np.zeros(4)
        self.sigma_f_snr = np.zeros(4)
        self.sigma_s_snr = np.zeros(4)
        self.std_f = np.zeros(4)
        for i in range(4):
            # Find min and max bandpass pixel indices:
            X_pix_min = np.where(min(l_orders[j[i]], key=lambda x:abs(x-(l_min[i])))==l_orders[j[i]])[0][0]
            X_pix_max = np.where(min(l_orders[j[i]], key=lambda x:abs(x-(l_max[i])))==l_orders[j[i]])[0][0]
            # Find number of pixels used in bandpass:
            n_pix_X = len(range(X_pix_min, X_pix_max))
            # Estimate flux uncertainty from each bandpass:
            f_snr_X = np.sum(f_snr_orders[j[i]][X_pix_min:X_pix_max]) / n_pix_X
            s_snr_X = np.sum(s_snr_orders[j[i]][X_pix_min:X_pix_max]) / n_pix_X        
            # Estimate S/N and uncertainties:
            self.sigma_f_snr[i] = 1/f_snr_X
            self.sigma_s_snr[i] = 1/s_snr_X
            self.f_snr_X[i] = f_snr_X
            self.s_snr_X[i] = s_snr_X
            # Estimate flat scatter:
            f      = f_orders[j[i]]/np.max(f_orders[j[i]])
            f_std0 = self.convolve(f, 'std',  2)
            f_std  = self.convolve(f_std0, 'mean', 100)
            self.std_f[i] = np.mean(f_std[X_pix_min:X_pix_max])
            # plt.figure()
            # plt.plot(l_orders[j[i]], f_std0, 'k-', linewidth=1.0, label=r'$\sigma_i$')
            # plt.plot(l_orders[j[i]], f_std,  'r-', linewidth=1.2, label=r'$\sigma_i/\mu_i$')
            # plt.show()
        # Uncertainty internally from spectrum:
        s_mea  = self.convolve(s, 'mean', 2)
        s_dif  = s/s_mea - 1
        s_std0 = self.convolve(s, 'std',  2)
        s_std  = self.convolve(s_std0, 'mean', 100)

        #----------------------------
        # ALL UNCERTAINTY CONSIDERED:
        #----------------------------
        # Shot Noise from flat blaze:
        self.sigma_f = np.sum(self.std_f) 
        # Three lines from order #57 and one from order #58 is used:
        self.sigma_w = ( 1/4*self.sigma_w[0] + 3/4*self.sigma_w[1] )/2
        # Find bandpass indices:
        _, _, bands = self.find_bandpass_fluxes(l, s, plot)
        V_indices,  R_indices  = bands[0], bands[1]
        K1_indices, H1_indices = bands[2], bands[3]
        Km_indices, Hm_indices = bands[4], bands[5]
        K2_indices, H2_indices = bands[6], bands[7]
        # Translate into flux uncertainty from each bandpass:
        self.std_V  = np.sum(s_std[V_indices])  / len(V_indices)  # Continuum bands
        self.std_R  = np.sum(s_std[R_indices])  / len(R_indices)
        self.std_K1 = np.sum(s_std[K1_indices]) / len(K1_indices) # Used for 1.09 Å square band fluxes
        self.std_H1 = np.sum(s_std[H1_indices]) / len(H1_indices)
        self.std_Km = np.sum(s_std[Km_indices]) / len(Km_indices) # Used for mean fluxes per wavelength
        self.std_Hm = np.sum(s_std[Hm_indices]) / len(Hm_indices)
        self.std_K2 = np.sum(s_std[K2_indices]) / len(K2_indices) # Used for triangular integrated fluxes
        self.std_H2 = np.sum(s_std[H2_indices]) / len(H2_indices) 
        self.sigma_bands = [s_std[K2_indices], s_std[H2_indices]] # Used for triangular norm fluxes
        # Combined uncertainties to be used for error propagation:
        x = self.sigma_w + self.sigma_f
        self.sigma_V = self.std_V + x
        self.sigma_R = self.std_R + x
        self.sigma_K1 = self.std_K1 + x; self.sigma_Km = self.std_Km + x; self.sigma_K2 = self.std_K2 + x
        self.sigma_H1 = self.std_H1 + x; self.sigma_Hm = self.std_Hm + x; self.sigma_H2 = self.std_H2 + x
        #--------------------------------------------------------------
        if plot is 1: pt.plot_sindex_scatter(l, s_dif, s_std0, s_std, self.bands)
        #--------------------------------------------------------------
        return s_std

    
    def find_bandpass_fluxes(self, l, s, plot=0):
        """
        This utility simply find the bandpass indices and fluxes given the bandpass widths.
        """
        # Shortwrite parameters:
        HK = self.HK_bandpass
        VR = self.VR_bandpass
        #------------------------------------------------
        # FIND INITIAL INDICES NEEDED FOR ALL BANDPASSES:
        #------------------------------------------------
        # Find central bandpass index:
        V_dex_cen = (np.abs(l-(self.V))).argmin()
        K_dex_cen = (np.abs(l-(self.K))).argmin()
        H_dex_cen = (np.abs(l-(self.H))).argmin()
        R_dex_cen = (np.abs(l-(self.R))).argmin()
        # Find wavelength indices for each bandpass:
        V_dex = (np.abs(l-(self.V+VR/2))).argmin() - (np.abs(l-(self.V-VR/2))).argmin()
        R_dex = (np.abs(l-(self.R+VR/2))).argmin() - (np.abs(l-(self.R-VR/2))).argmin()
        # Find wavelength indices for 1.09 Å bandpass:
        K1_dex = (np.abs(l-(self.K+HK/2))).argmin() - (np.abs(l-(self.K-HK/2))).argmin()
        H1_dex = (np.abs(l-(self.H+HK/2))).argmin() - (np.abs(l-(self.H-HK/2))).argmin()
        # Find wavelength indices for 2 x 1.09 Å lower widths:
        K2_dex = (np.abs(l-(self.K+HK))).argmin() - (np.abs(l-(self.K-HK))).argmin()
        H2_dex = (np.abs(l-(self.H+HK))).argmin() - (np.abs(l-(self.H-HK))).argmin()
        # Select only even bandswidths:
        for VR_dex_i in [V_dex, R_dex]:
            if VR_dex_i % 2 == 0: VR_dex = int(VR_dex_i / 2)
            else: VR_dex = int((VR_dex_i - 1) / 2) 
        for HK1_dex_i in [K1_dex, H1_dex]:
            if HK1_dex_i % 2 == 0: HK1_dex = int(HK1_dex_i / 2)
            else: HK1_dex = int((HK1_dex_i - 1) / 2) 
        for HK2_dex_i in [K2_dex, H2_dex]:
            if HK2_dex_i % 2 == 0: HK2_dex = int(HK2_dex_i / 2)
            else: HK2_dex = int((HK2_dex_i - 1) / 2) 
        # Find square bandpass indices:
        V_indices = np.arange(V_dex_cen-VR_dex, V_dex_cen+VR_dex)
        R_indices = np.arange(R_dex_cen-VR_dex, R_dex_cen+VR_dex)
        # Covert to a full-range indices:
        K1_indices = np.arange(K_dex_cen-HK1_dex, K_dex_cen+HK1_dex)
        H1_indices = np.arange(H_dex_cen-HK1_dex, H_dex_cen+HK1_dex)
        # Covert to a full-range indices:
        K2_indices = np.arange(K_dex_cen-HK2_dex, K_dex_cen+HK2_dex+1)
        H2_indices = np.arange(H_dex_cen-HK2_dex, H_dex_cen+HK2_dex+1)

        #--------------------------------------
        # DEFINE TRIANGULAR H AND K BANDPASSES:
        #--------------------------------------
        # Split out index ranges:
        k1_indices = np.arange(K_dex_cen-HK2_dex, K_dex_cen)
        k2_indices = np.arange(K_dex_cen, K_dex_cen+HK2_dex+1)
        h1_indices = np.arange(H_dex_cen-HK2_dex, H_dex_cen)
        h2_indices = np.arange(H_dex_cen, H_dex_cen+HK2_dex+1)
        # Compute linear relations on either side triangle:
        coefs_k1 = np.polyfit([K_dex_cen-HK2_dex, K_dex_cen], [0, 1], 1); poly_k1 = np.poly1d(coefs_k1)
        coefs_k2 = np.polyfit([K_dex_cen, K_dex_cen+HK2_dex], [1, 0], 1); poly_k2 = np.poly1d(coefs_k2)
        coefs_h1 = np.polyfit([H_dex_cen-HK2_dex, H_dex_cen], [0, 1], 1); poly_h1 = np.poly1d(coefs_h1)
        coefs_h2 = np.polyfit([H_dex_cen, H_dex_cen+HK2_dex], [1, 0], 1); poly_h2 = np.poly1d(coefs_h2)
        # Find count values for each line:
        s_tri_k1 = poly_k1(k1_indices)
        s_tri_k2 = poly_k2(k2_indices)
        s_tri_h1 = poly_h1(h1_indices)
        s_tri_h2 = poly_h2(h2_indices)
        # Combine triangular count values:
        s_tri_K = np.append(s_tri_k1, s_tri_k2)
        s_tri_H = np.append(s_tri_h1, s_tri_h2)       
        # Define finer regular grid:
        l_k1_grid = np.linspace(l[k1_indices[0]], l[k1_indices[-1]], 1e4)
        l_k2_grid = np.linspace(l[k2_indices[0]], l[k2_indices[-1]], 1e4)
        l_h1_grid = np.linspace(l[h1_indices[0]], l[h1_indices[-1]], 1e4)
        l_h2_grid = np.linspace(l[h2_indices[0]], l[h2_indices[-1]], 1e4)
        # Interpolate triangular data function:
        s_tri_k1_grid = scipy.interpolate.griddata(l[k1_indices], s_tri_k1, l_k1_grid, method='linear')
        s_tri_k2_grid = scipy.interpolate.griddata(l[k2_indices], s_tri_k2, l_k2_grid, method='linear')
        s_tri_h1_grid = scipy.interpolate.griddata(l[h1_indices], s_tri_h1, l_h1_grid, method='linear')
        s_tri_h2_grid = scipy.interpolate.griddata(l[h2_indices], s_tri_h2, l_h2_grid, method='linear')
        # Interpolate spectral data:
        s_spc_k1_grid = scipy.interpolate.griddata(l[k1_indices], s[k1_indices], l_k1_grid, method='linear')
        s_spc_k2_grid = scipy.interpolate.griddata(l[k2_indices], s[k2_indices], l_k2_grid, method='linear')
        s_spc_h1_grid = scipy.interpolate.griddata(l[h1_indices], s[h1_indices], l_h1_grid, method='linear')
        s_spc_h2_grid = scipy.interpolate.griddata(l[h2_indices], s[h2_indices], l_h2_grid, method='linear')
        # Find wavelength value of intersection:
        dex_k1_inter = np.where(np.abs(s_spc_k1_grid - s_tri_k1_grid) < 5e-4)[0][0]
        dex_k2_inter = np.where(np.abs(s_spc_k2_grid - s_tri_k2_grid) < 5e-4)[0][-1]
        dex_h1_inter = np.where(np.abs(s_spc_h1_grid - s_tri_h1_grid) < 5e-4)[0][0]
        dex_h2_inter = np.where(np.abs(s_spc_h2_grid - s_tri_h2_grid) < 5e-4)[0][-1]
        # Find coordinates of intersection points:
        l_k1_inter, s_k1_inter = l_k1_grid[dex_k1_inter], s_tri_k1_grid[dex_k1_inter]
        l_k2_inter, s_k2_inter = l_k2_grid[dex_k2_inter], s_tri_k2_grid[dex_k2_inter]
        l_h1_inter, s_h1_inter = l_h1_grid[dex_h1_inter], s_tri_h1_grid[dex_h1_inter]
        l_h2_inter, s_h2_inter = l_h2_grid[dex_h2_inter], s_tri_h2_grid[dex_h2_inter]
        # Find closest wavelength index to intersection point:
        dex_spc_k1_inter = (np.abs(l_k1_inter - l[k1_indices])).argmin()
        dex_spc_k2_inter = (np.abs(l_k2_inter - l[k2_indices])).argmin()
        dex_spc_h1_inter = (np.abs(l_h1_inter - l[h1_indices])).argmin()
        dex_spc_h2_inter = (np.abs(l_h2_inter - l[h2_indices])).argmin()
        
        #------------------
        # FIND FLUX VALUES:
        #------------------
        # Find fluxes in each continuum passbands:
        V_fluxes  = s[V_indices] 
        R_fluxes  = s[R_indices]
        # Find final count values for retangular 1.09Å filter:
        K1_fluxes = s[K1_indices]
        H1_fluxes = s[H1_indices]
        # Find final count values for mean filter:
        Km_indices = np.arange(k1_indices[dex_spc_k1_inter], k2_indices[dex_spc_k2_inter])
        Hm_indices = np.arange(h1_indices[dex_spc_h1_inter], h2_indices[dex_spc_h2_inter])
        Km_fluxes = s[Km_indices]
        Hm_fluxes = s[Hm_indices] 
        # Find final count values for triangular filter:
        k1_fluxes = s_tri_k1[:dex_spc_k1_inter].tolist()
        h1_fluxes = s_tri_h1[:dex_spc_h1_inter].tolist()
        k2_fluxes = s_tri_k2[dex_spc_k2_inter:].tolist()
        h2_fluxes = s_tri_h2[dex_spc_h2_inter:].tolist()
        K2_fluxes = np.array(k1_fluxes + Km_fluxes.tolist() + k2_fluxes)
        H2_fluxes = np.array(h1_fluxes + Hm_fluxes.tolist() + h2_fluxes)
        # Construct coordinate array for precise polygon-area calculation:
        Kp_wave = np.array([l[K2_indices[0]], l_k1_inter] + l[Km_indices].tolist() + \
                              [l_k2_inter, l[K2_indices[-1]] ])
        Hp_wave = np.array([l[H2_indices[0]], l_h1_inter] + l[Hm_indices].tolist() + \
                              [l_h2_inter, l[H2_indices[-1]] ])
        Kp_fluxes = np.array([0, s_k1_inter] + Km_fluxes.tolist() + [s_k2_inter, 0])
        Hp_fluxes = np.array([0, s_h1_inter] + Hm_fluxes.tolist() + [s_h2_inter, 0])
        Kp_coors = np.array([Kp_wave, Kp_fluxes])
        Hp_coors = np.array([Hp_wave, Hp_fluxes])
        # Combine to return:
        band_fluxes  = [V_fluxes, R_fluxes, K1_fluxes, H1_fluxes, Km_fluxes, Hm_fluxes, \
                        K2_fluxes, H2_fluxes, Kp_coors, Hp_coors]
        tri_function = [s_tri_K, s_tri_H]
        band_indices = [V_indices, R_indices, K1_indices, H1_indices, Km_indices, Hm_indices, \
                        K2_indices, H2_indices, K_dex_cen, H_dex_cen]
        #--------------------------------------------------------------
        if plot is 1:
            pt.plot_sindex_bands(l, s, s_tri_K, s_tri_H, K2_indices, H2_indices, K2_fluxes, H2_fluxes, \
                                 l_k1_inter, l_k2_inter, l_h1_inter, l_h2_inter, \
                                 s_k1_inter, s_k2_inter, s_h1_inter, s_h2_inter, \
                                 Kp_wave, Hp_wave, Kp_fluxes, Hp_fluxes, Km_indices, Hm_indices, \
                                 self.K, self.H, K1_indices, H1_indices)
            pt.plot_sindex_fluxes(l, s, band_indices, band_fluxes, self.bands)
        #--------------------------------------------------------------
        return band_fluxes, tri_function, band_indices


    def sindex(self, flux_results, l, save=None):
        """ 
        This utility is a general function to calculate the S index in (1) the standard way and (2) using 
        mean bandpass fluxes if 'dex_meanband' is available.
        """
        # Unpack flux results:
        bandfluxes, tri_func, bandindices = flux_results[0], flux_results[1], flux_results[2]
        # Find V an R bandpass fluxes: 
        val_V = np.sum(bandfluxes[0]); V = ufloat(val_V, val_V*self.sigma_V)
        val_R = np.sum(bandfluxes[1]); R = ufloat(val_R, val_R*self.sigma_R)
        #--------------------------------------------
        # FIND S INDEX FROM 1.09 Å INTEGRATED FLUXES:
        #--------------------------------------------
        val_K1 = np.sum(bandfluxes[2]); K1 = ufloat(val_K1, val_K1*self.sigma_K1)
        val_H1 = np.sum(bandfluxes[3]); H1 = ufloat(val_H1, val_H1*self.sigma_H1)
        # Calculate S index:
        sindex_HK1 = 8 * (H1 + K1)/(R + V) *2.4
        #--------------------------------------------
        # FIND S INDEX FROM MEAN FLUX PER WAVELENGTH:
        #--------------------------------------------
        val_Vm = np.mean(bandfluxes[0]); Vm = ufloat(val_Vm, val_Vm*self.sigma_V)
        val_Rm = np.mean(bandfluxes[1]); Rm = ufloat(val_Rm, val_Rm*self.sigma_R)
        val_Km = np.mean(bandfluxes[6]); Km = ufloat(val_Km, val_Km*self.sigma_Km) 
        val_Hm = np.mean(bandfluxes[7]); Hm = ufloat(val_Hm, val_Hm*self.sigma_Hm)
        # val_Km = np.mean(bandfluxes[4]); Km = ufloat(val_Km, val_Km*self.sigma_Km) 
        # val_Hm = np.mean(bandfluxes[5]); Hm = ufloat(val_Hm, val_Hm*self.sigma_Hm)
        # Calculate S index:
        sindex_HKm = 8 * (Hm + Km)/(Rm + Vm) * self.HK_bandpass/self.VR_bandpass * 2.4
        #------------------------------------------------------
        # FIND S INDEX FROM TRIANGULAR BANDSPASS NORMALIZATION:
        #------------------------------------------------------
        val_Kn = np.sum(bandfluxes[6] * tri_func[0])
        val_Hn = np.sum(bandfluxes[7] * tri_func[1])
        sigma_Kn = np.sum(self.sigma_bands[0] * tri_func[0])
        sigma_Hn = np.sum(self.sigma_bands[1] * tri_func[1]) 
        Kn = ufloat(val_Kn, val_Kn*sigma_Kn)
        Hn = ufloat(val_Hn, val_Hn*sigma_Hn)
        # Calculate S index:
        sindex_HKn = 8 * (Hn + Kn)/(R + V) *2.4

        #------------------------------------------------
        # FIND S INDEX FROM TRIANGULAR INTEGRATED FLUXES:
        #------------------------------------------------
        val_K2 = np.sum(bandfluxes[6]); K2 = ufloat(val_K2, val_K2*self.sigma_K2)
        val_H2 = np.sum(bandfluxes[7]); H2 = ufloat(val_H2, val_H2*self.sigma_H2)
        # Calculate S index:
        sindex_HK2 = 8 * (H2 + K2)/(R + V)
        #----------------------------------------------
        # FIND S INDEX FROM INCLOSED POLYGON FLUX AREA:
        #----------------------------------------------
        # Unpack, reverse, add starting point:
        l_V = l[bandindices[0]].tolist(); lr_V = l_V[::-1]; lr_V = [l_V[0]] + [l_V[-1]] + lr_V
        l_R = l[bandindices[1]].tolist(); lr_R = l_R[::-1]; lr_R = [l_R[0]] + [l_R[-1]] + lr_R
        l_K = bandfluxes[8][0].tolist();  lr_K = l_K[::-1]; lr_K = [l_K[0]] + lr_K
        l_H = bandfluxes[9][0].tolist();  lr_H = l_H[::-1]; lr_H = [l_H[0]] + lr_H
        s_V = bandfluxes[0].tolist();     sr_V = s_V[::-1]; sr_V = [0] + [0] + sr_V 
        s_R = bandfluxes[1].tolist();     sr_R = s_R[::-1]; sr_R = [0] + [0] + sr_R
        s_K = bandfluxes[8][1].tolist();  sr_K = s_K[::-1]; sr_K = [0] + sr_K
        s_H = bandfluxes[9][1].tolist();  sr_H = s_H[::-1]; sr_H = [0] + sr_H
        #  list for calculation: 
        val_Vp = self.polygon_area(np.array([lr_V, sr_V]).T)
        val_Kp = self.polygon_area(np.array([lr_K, sr_K]).T)
        val_Hp = self.polygon_area(np.array([lr_H, sr_H]).T)
        val_Rp = self.polygon_area(np.array([lr_R, sr_R]).T)
        
        # plt.figure()
        # plt.plot(lr_V, sr_V, 'b-')
        # plt.plot(lr_V, sr_V, 'r*')
        # plt.show()
        # Combine with uncertainty:
        Vp = ufloat(val_Vp, val_Vp*self.sigma_V)
        Kp = ufloat(val_Kp, val_Kp*self.sigma_K2)
        Hp = ufloat(val_Hp, val_Hp*self.sigma_H2)
        Rp = ufloat(val_Rp, val_Rp*self.sigma_R)
        # Calculate S index:
        sindex_HKp = 8 * (Hp + Kp)/(Rp + Vp)
        #--------------------------------------------------------------
        if save is 0:
            self.s1 = ['1:', val_V,  val_K1, val_H1, val_R,  sindex_HK1]
            self.sn = ['n:', val_V,  val_Kn, val_Hn, val_R,  sindex_HKn]
            self.sm = ['m:', val_Vm, val_Km, val_Hm, val_Rm, sindex_HKm]
            self.s2 = ['2:', val_V,  val_K2, val_H2, val_R,  sindex_HK2]
            self.sp = ['p:', val_Vp, val_Kp, val_Hp, val_Rp, sindex_HKp]
        return np.array([sindex_HK1, sindex_HKn, sindex_HKm, sindex_HK2, sindex_HKp])
        
        
    def polygon_area(self, xy, plot=0):
        """ 
        This utility takes coordinates (x, y) ordered in an array and calculates the polygon area enclosed.
        The coordinates needs to be ordered in a counter-clock-wise manner since the circuference using
        Green's theorem is used to equate the polygon area.
        """
        l = len(xy)
        s = 0.0
        for i in range(l):
            j = (i+1)%l  # keep index in [0,l)
            s += (xy[j,0] - xy[i,0])*(xy[j,1] + xy[i,1])
        #--------------------------------------------------------------
        return -0.5*s
    
    
    def results(self):
        print('################################################')
        print('             {} - {}        '.format(self.target, self.date))
        print('################################################')
        head_SF = self.hdul[self.SF_dex[0]][0].header
        head_FF = self.hdul[self.FF_dex[0]][0].header
        print('Magnitude = {},  Seeing = {}'.format(self.magnitude, self.seeing))
        print('Exptime flat:  t = {} s'.format(head_SF['EXPTIME']))
        print('Exptime star:  t = {} s'.format(head_FF['EXPTIME']))
        print('------------------------------------------------')
        print('              CCD NOISE PROPERTIES              ')
        print('------------------------------------------------')
        BF_mean, BF_std = np.mean(self.BF), np.std(self.BF)
        DF_mean, DF_std = np.mean(self.DF), np.std(self.DF)
        print('Bias master : mean = {:.4g}, std = {:.4g}'.format(BF_mean, BF_std))
        print('Dark current: mean = {:.4g}, std = {:.4g}'.format(DF_mean, DF_std))
        print('GAIN = {:.3g} e-/ADU'.format(self.gain))
        print('RON  = {:.3g} ADU'.format(BF_std))
        print('VAR  = {:.3g} ADU (=<RON^2>)'.format(BF_std**2))
        print('------------------------------------------------')
        print('             BACKGROUND SKY & SCATTER           ')
        print('------------------------------------------------')
        print('Flat mean background counts: {:.1f}'.format(self.f_flux_sky))
        print('Star mean background counts: {:.1f}'.format(self.s_flux_sky))
        print('------------------------------------------------')
        print('                  RV CORRECTION                 ')
        print('------------------------------------------------')
        print('Barycentric RV correction: {:.2f} km/s'.format(self.delta_v_baryc))
        print('Star motion RV Correction: {:.2f} km/s'.format(self.rv_amp))
        print('Correction in velocity   : {:.2f} km/s'.format(self.delta_v))
        print('Correction in wavelength : {:.2f} Å'.format(self.delta_l))
        print('Correction in pixelspace : {:.2f}'.format(self.delta_p))
        print('------------------------------------------------')
        print('               SNR & UNCERTAINTIES              ')
        print('------------------------------------------------')
        f_snr_max, s_snr_max = self.f_snr_max, self.s_snr_max
        print('S/N in order #57:  {:.1f} (flat),  {:.1f} (star)'.format(f_snr_max[1], s_snr_max[1]))
        print('S/N in order #58:  {:.1f} (flat),  {:.1f} (star)'.format(f_snr_max[0], s_snr_max[0]))
        print('------------------------------------------------')
        snx = [self.s_snr_X[0], self.s_snr_X[1], self.s_snr_X[2], self.s_snr_X[3]]
        snr =[self.sigma_s_snr[0]*100,self.sigma_s_snr[1]*100,self.sigma_s_snr[2]*100,self.sigma_s_snr[3]*100]
        std = [self.sigma_V*100, self.sigma_K1*100, self.sigma_H1*100, self.sigma_R*100]
        print('Bandpass  :  V      K      H      R      | Total')
        print('S/N       :  {:.3g}   {:.3g}   {:.3g}   {:.3g}   |'.format(snx[0], snx[1], snx[2], snx[3]))
        print('sigma(S/N):  {:.3g}%  {:.3g}%  {:.3g}%  {:.3g}%  | {:.1f}%'.format(snr[0], snr[1], snr[2],\
                                                                                  snr[3], np.sum(snr)))
        print('sigma(std):  {:.3g}%  {:.3g}%  {:.3g}%  {:.3g}%  | {:.1f}%'.format(std[0], std[1], std[2],\
                                                                                  std[3], np.sum(std)))
        print('sigma(wav):                              | {:.2f}%'.format(self.sigma_w*100))
        print('sigma(fla):                              | {:.2f}%'.format(self.sigma_f*100))
        print('------------------------------------------------')
        print('                    S INDEX                     ')
        print('------------------------------------------------')
        print(self.s1)
        print(self.sn)
        print(self.sm)
        print(self.s2)
        print(self.sp)
        print('------------------------------------------------')
            
    
    ########################################################################################################
    #                                            OPTIMAL WIDTHS                                            #
    ########################################################################################################

    def find_optimal_width(self, image=None, trace=None, plot=0):
        """
        This utility takes most preferably a reduced flat image and the polynomial describtion traced,
        and first cut out a bandpass defined by disp_lenght and cross_width. Looping through increasing
        spatials widths the S/N ratio is found for each, and the spatial width asigned to the highest
        S/N ratio is optimal for linear extraction. To return the results in terms of FWHM a Gauss function
        is fitted to the spatial width of maximum flux.  
        """
        # Check if 'image' and 'trace' is defined:
        if image==None: image = self.F_calib
        if trace==None: trace = self.trace
        
        # Cut out order:
        widths = np.arange(1, 40)
        order = self.cut_out_order(image, np.polyval(trace['order_2'], self.disp), widths[-1])

        # Find maximum of blaze function:
        blaze     = order.sum(axis=1)
        blaze_max = np.max(blaze)
        index_max = np.nanargmax(blaze)

        # Find mean sky background along disp direction used for S/N ratio:
        flux_inter, _ = self.mean_background(image, trace, plot=0)
        
        # Loop over spatial widths:
        snr = np.zeros(len(widths))
        for w in widths:
            order_w    = order[index_max, widths[-1]-1-w:widths[-1]-1+w]
            flux_order = np.sum(order_w) 
            snr[w-1]   = self.signal_to_noise(flux_order, len(order_w), flux_order)
            
        # Find highest S/N ratio optimal order width:
        index_max_snr       = np.argmax(snr)
        optimal_order_width = widths[index_max_snr]

        # Find residual inter-order width:
        order_distance = int(((self.ref_cen_pos[1] - self.ref_cen_pos[2]) + \
                              (self.ref_cen_pos[2] - self.ref_cen_pos[3]))/2)
                             
        #optimal_inter_order_width = int(order_distance - 2.5*optimal_order_width)

        #--------------------------------------------------------------
        if plot is 1:
            pt.plot_optimal_width(widths, order, blaze_max, index_max, flux_inter, snr, optimal_order_width)
        #--------------------------------------------------------------
        self.order_width = optimal_order_width
        #--------------------------------------------------------------
        return self.order_width

    
    def mean_background(self, image, trace, plot=0):
        """
        This utility use 'trace' and 'cut_out_order' to select the pixel sky-background in a bandpass on
        both sides of the order of interest. In spatial direction on each side the median pixel value is
        found, and lastly the mean value of each side is then computed. Returned is a 1D spectrum describing
        the background (e.g. used by the 'signal_to_noise' utility).
        """
        # Find midpoint of inter orders:
        midpoint_below = (self.ref_cen_pos[1] - self.ref_cen_pos[2])/2
        midpoint_above = (self.ref_cen_pos[2] - self.ref_cen_pos[3])/2

        # Move fit to the midpoint of inter orders:
        yfit_below = np.polyval(trace['order_1'], self.disp) + np.ones(len(self.disp))*midpoint_below
        yfit_above = np.polyval(trace['order_2'], self.disp) + np.ones(len(self.disp))*midpoint_above
        yfit_order = np.polyval(trace['order_2'], self.disp) + np.ones(len(self.disp))

        # Set cross width for background cut to half the distance between orders:
        # (here the position of the order is a limitation)
        cross_order_width = math.floor(yfit_below[0])*2 - 1
        # (else if order are moved up use)
        #cross_order_width = int((self.ref_cen_pos[1] - self.ref_cen_pos[2])[0]/2 - 1)

        # Cut out stellar background on both sides:
        back_below = self.cut_out_order(image, yfit_below, cross_order_width)
        back_above = self.cut_out_order(image, yfit_above, cross_order_width)

        # Sum order to 1D spectrum and mean them:
        l_sky = (np.median(back_below, axis=1) + np.median(back_above, axis=1))/2.
        flux_sky_mean = abs(l_sky.mean())
        
        #-----------------------------------------------------------:
        if plot is 1: pt.plot_sky_background(image, self.disp, yfit_below, yfit_above, yfit_order, l_sky)
        #--------------------------------------------------------------
        return flux_sky_mean, l_sky    

    
    def signal_to_noise(self, flux_star, n_pix_star, flux_sky):
        """
        This function calculates the S/N ratio using the 1D spectrum of the object and sky-background.
        Purely by statistics with and increasing number of pixel used to define the object 'n_pix_object',
        the S/N ratio will decrease. The noise sources describing a CCD are the 'gain' (e-/ADU) and 'ron',
        read-out-noise (e-). 
        """
        # See Schroeder (1999) p. 317 or Bradt (2004) p. 163:
        signal = flux_star*self.gain 
        noise  = np.sqrt(flux_star*self.gain + flux_sky*self.gain*n_pix_star + self.ron*n_pix_star)
        #--------------------------------------------------------------
        return signal / noise
   
    ########################################################################################################
    #                          GENERAL UTILITIES SPECIALIZED TO THIS SOFTWARE                              #
    ######################################################################################################## 
        
    def blue_moves(self, path, plot=0):
        """ 
        This routine measures the drift of the spectrum over time by using ThAr lines in the same order 
        as the Ca II H & K lines. (Fun fact: the software name comes from 'Blue Moves' which is the eleventh
        studio album release by Elton John, released in October 1976. 
        """
        # Load all files from same folder:
        img_files = np.sort(glob.glob('{}{}*'.format(path, self.img_name)))
        hdu       = np.array([fits.open(str(files)) for files in img_files])
        n         = len(img_files)
        
        # Find time scaling to utc time and Julian Date
        time = [hdu[i][0].header['JD-DATE'] for i in range(n)]
        
        # Loop through all ThAr images:
        move_x  = np.zeros(n)
        move_y  = np.zeros(n)
        sigma_x = np.zeros(n-1)
        sigma_y = np.zeros(n-1)
        for i in range(n):
            
            # Open and close one image at a time:
            with fits.open(str(img_files[i])) as hdu_i:

                # Select focused spectral region:
                T_i = hdu_i[0].data[300:480, 420:2270].T

                # UTILITY CALL: Locate coordinates of lines:
                COF_i, _, _ = self.peak_finder(T_i, sigma=5, plot=0)

                # UTILITY CALL: Remove lines too close to borders:
                COF_i, N_lines = self.image_border(T_i, COF_i)

                # UTILITY CALL: Only use same lines each time:
                if i==0:
                    #COF_0, _, _ = self.peak_finder(T_i, sigma=5, plot=0)
                    COF_0 = COF_i
                if i is not 0:
                    indices0, indices1 = self.match_coordinates(COF_0, COF_i, threshold=5, plot=1)
                    # Find scatter of the drift for each line:
                    if i > 1:
                        diff_x = COF_i[indices1,0] - x
                        diff_y = COF_i[indices1,1] - y
                        sigma_x[i-1] = np.std(diff_x)
                        sigma_y[i-1] = np.std(diff_y)
                    # Find coordinates (x and y needs to be after if < 1 statement):
                    x = COF_i[indices1,0]
                    y = COF_i[indices1,1]
                    move_x[i] = x.mean()
                    move_y[i] = y.mean()
                    
                    # Print to bash:
            pt.compilation(i, n, 'Blue Moves')
        print

        # Convert to relative changes:
        move_x = move_x[1::] - move_x[1::].mean()
        move_y = move_y[1::] - move_y[1::].mean()
        time   = time[1::]
        #-----------------------------------------------------------
        if plot is 1:
            np.savetxt('{}bluemoves.txt'.format(self.path), np.vstack([time, move_y, sigma_y]).T)
            pt.plot_rv_stability(time, move_y, sigma_y)
        #-----------------------------------------------------------
        return

    
    def image_border(self, image, pixel_coor, border_edge=20):
        """
        This utility takes an array of pixel coordinates and finds coordinates that is closer than 20 pixels
        to the image 'border_edge'. These coordinates are then removed from the array and a new array,
        'new_pixel_coor', is returned together with the new (lower) number of coordinates 'N_coor'.
        """
        # Unpack pixel coordinates:
        x = pixel_coor[:,0]
        y = pixel_coor[:,1]
        # Check if stellar coordinates are too close to borders:
        i_x1 = np.where(x < border_edge)[0]
        i_y1 = np.where(y < border_edge)[0]
        i_x2 = np.where(x > np.shape(image)[0]-border_edge)[0]
        i_y2 = np.where(y > np.shape(image)[1]-border_edge)[0]
        i_xy = np.hstack([i_x1, i_x2, i_y1, i_y2])
        # Discard these coordinates:
        x_new = np.delete(x, i_xy)
        y_new = np.delete(y, i_xy)
        N_coor = len(x)
        #-----------------------------------------------------------
        return np.array([x_new, y_new]).T, N_coor

    ########################################################################################################
    #                                     GENERAL STRUCTURAL ALGORITHMS                                    #
    ########################################################################################################

    def peak_finder(self, pixel_array, min_pix=7, sigma=1, plot=0):       
        """
        This utility takes a pixel array and use the 'scipy.ndimage' package to find local maxima within an
        image. These are determined upon the number of standard deviations, 'sigma', and a minimum of pixels,
        'min_pix', a structure should be considered. From the returned structure the same package determines
        the Center Of Flux ('COF') in coordinate space (x, y), and the circular 'radius' for each, together
        with the number of local maximum structures, 'N_struct', detected within the pixel_array.
        """
        # FIND CENTER OF FLUX FOR STARS ABOVE THRESHOLD:
        # Define threshold as a number of standard deviations above the mean:
        threshold = np.mean(pixel_array) + sigma*np.std(pixel_array)
        # Find all pixels above the threshold:
        above_threshold = np.where(pixel_array > threshold, 1, 0)
        # Label the structures (where starmap = 1 that are adjacent to others):
        labels, N_structs = scipy.ndimage.label(above_threshold, structure = np.ones((3,3)))
        # Sum the number of elements in each structure:
        sums = scipy.ndimage.sum(above_threshold, labels, range(1,N_structs+1))
        # Choose only structures with more than min_pix elements (+1 for index mismatch):
        structs = np.where(sums > min_pix)[0] + 1
        # Define starmap as 0 where there are no stars and 1 where there are stars:
        struct_map = np.zeros(np.shape(pixel_array))
        for struct in structs:  struct_map = struct_map + np.where(labels == struct, 1, 0)
        # Label all the structures again:
        labels, N_structs = scipy.ndimage.label(struct_map, structure = np.ones((3,3)))
        # Find the center of flux of all the structures found above threshold:
        COF = scipy.ndimage.center_of_mass(pixel_array, labels, range(1, N_structs+1))
        # Estimate the radius of the structures in pixels:
        radius = np.sqrt(sums[structs-1]/np.pi) 
        # From tuple to array:
        COF = np.asarray(COF)
        #--------------------------------------------------------------
        if plot is 1: # NEEDS ACTIVATION FROM SOURCE
            plt.figure()
            plt.imshow(pt.linear(pixel_array.T), cmap='Blues', origin='lower')
            plt.scatter(COF[:,0], COF[:,1], s=radius*12, facecolors='none', edgecolors='r', marker='s')
            plt.show()
        #--------------------------------------------------------------
        return COF, radius, N_structs


    def find_peak_in_noise(self, s, peak_coor, plot=0):
        """
        This utility identifies the 
        """
        # Define limits for peak search:
        limits = [int(peak_coor-25), int(peak_coor+25)]
        
        # Different conditions:
        conv  = self.res_power * 1e-3   # Smooth-filter scale linear with resolving power
        width = self.len_disp/10        # Width scales likewise with the pixel scale
        
        # Find all peaks: 
        peaks_all_dex, _ = scipy.signal.find_peaks(s)
        peaks_all_val    = s[peaks_all_dex]

        # Find all approx peaks from convolved spectrum:
        s_conv      = self.convolve(s, 'median', int(conv))
        s_conv      = self.convolve(s_conv, 'mean',  int(conv))
        peaks_conv_dex, _ = scipy.signal.find_peaks(s_conv)
        peaks_conv_val    = s_conv[peaks_conv_dex]

        # Select peaks inside disp limits range:
        ndarray = (peaks_conv_dex > limits[0]) * (peaks_conv_dex < limits[1])
        peaks_limit_dex = peaks_conv_dex[ndarray]
        peaks_limit_val = peaks_conv_val[ndarray]

        # Find x highest peaks from convolved spectrum:
        peak_conv_dex = heapq.nlargest(1, np.arange(len(peaks_limit_val)), key=peaks_limit_val.__getitem__)[0]
        peak_conv_pix = peaks_limit_dex[peak_conv_dex]
        
        # Make bold array around peak with dobbelt the width of conv:
        peak_dex = (peaks_all_dex>peak_conv_pix-conv)*(peaks_all_dex<peak_conv_pix+conv)

        # Select peak:
        peak = peaks_all_dex[peak_dex][np.argmax(peaks_all_val[peak_dex])]
        
        #--------------------------------------------------------------
        if plot is 1:
            pt.plot_arc_peak(s, s_conv, peaks_limit_dex, peaks_limit_val, peaks_all_dex, \
                                peaks_all_val, peak, limits)
        #--------------------------------------------------------------
        return peak

    
    def convolve(self, data0, filtertype, n): 
        """
        This function can be used to correct for slow trends using e.g. a "moving mean" filter. The utility
        takes the flatten data, a string with the desired filter, and n number of points is should smooth
        the data with. Compared to the bottleneck package this function do not leave a offset.
        """
        # Constants:
        data  = data0.copy()           # Avoid overwritting data:
        data_new = np.zeros(len(data)) # To pick up new data
        nzero = np.zeros(2*n+1)        # optimization constant
        # Available filters:
        if filtertype=='mean':   moving_filter = np.mean
        if filtertype=='median': moving_filter = np.median
        if filtertype=='sum':    moving_filter = np.sum
        if filtertype=='std':    moving_filter = np.std
        # Interval: d[n, 1+n, ... , N-1, N-n]
        for i in range(len(data)-2*n):   
            data_new[n+i] = moving_filter(data[range((n+i)-n, (n+i)+n+1)])
        for i in range(n):
            # Interval: d[-n, -(n-1), ... , n-1, n] - Low end of data
            low = nzero
            low[range(n-i)] = data[0]*np.ones(n-i)
            low[-(n+1+i):]  = data[range(0, n+1+i)]
            data_new[i]     = moving_filter(low)
            # Interval: d[N-n, N-(n-1), ... , N+(n-1), N+n] - High end of data
            high = nzero
            high[range(n+1+i)] = data[range(len(data)-(n+i+1), len(data))]
            high[-(n-i):]      = data[-1]*np.ones(n-i)
            data_new[len(data)-1-i] = moving_filter(high)
        #--------------------------------------------------------------
        return data_new
  
    
    def match_coordinates(self, array1, array2, threshold=10, plot=0):
        """
        This function match two set of coordinates. This is done by a purely geometrical technique and
        looking at the histogram. It finds the minimum distance from i'th array1 star to every other array2
        star. Here indices is the rows with all the indices of matching. To select only common coordinates,
        then use 'indices2'.
        """
        # Placeholders:
        value_min = np.zeros(len(array1))
        index_min = {}
        # FIND MINIMUM DISTANCE WITH PYTHAGOREAN GEOMETRY: 
        for i in range(len(array1)):
            d = np.sqrt( (array2[:,0]-array1[i,0])**2 + (array2[:,1]-array1[i,1])**2 )
            index_min[i] = np.argmin(d)
            value_min[i] = d[index_min[i]]
        # array1 indices of all stars:
        index_min = list(index_min.values())
        # find array1 stars within threshold:
        indices1 = np.where(value_min<threshold)[0]
        # Final list of matching array2 stars:
        indices2 = [index_min[i] for i in indices1]
        #-----------------------------------------------------------
        if plot==1: pt.plot_match_coordinates(array1, array2, indices1, indices2)
        #-----------------------------------------------------------
        return indices1, indices2
    

    def locate_outliers(self, S, convolve_step=3, cutoff=1e-3, plot=0): 
        """
        This function can be used to locate bad data points using a "moving median" filter. For the median 
        filter, instead of deleting bad data these are replaced by a median value. It takes the data series
        "signal" (S), and integer "convolve_step" used as step size of moving filter, a 'cutoff' integer
        limits for the bad data, and returns the the signal "SS" corrected for bad data.
        """
        # Data:
        x = range(len(S))
        SS = S.copy()
        # Finding dif:
        S_med = self.convolve(S, 'median', convolve_step)
        #S_med = self.convolve(S, 'median', convolve_step)
        dif0  = S/S_med - 1
        if cutoff is None: cutoff = np.std(dif0)
        # Replace median signal if outside cutoff region:
        above     = np.where(dif0>+cutoff)[0]
        SS[above] = S_med[above]
        below     = np.where(dif0<-cutoff)[0]
        SS[below] = S_med[below]
        # Consistency check for median replacement:
        SS_med = self.convolve(SS, 'median', convolve_step)
        dif1  = SS/SS_med - 1
        #--------------------------------------------------------------
        if plot==1: pt.plot_locate(x, dif0, dif1, cutoff, convolve_step, above, below)
        #--------------------------------------------------------------
        return SS 
