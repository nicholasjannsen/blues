"""
CLASS DESCRIBTION:
------------------
This is the astronomical pipline used to reduce and obtain photometry for the AUSAT (the first satellite 
lunched by Aarhus University). This pipeline was constructed as a final project for the 2nd workshop of 
Delphini-1. The overall aim for the routines in this DELPHINI python class, is to perform 'image reduction' 
and 'photmetry'.To each routine a description for its functionality is given. For the functionality see the
README file downloaded alongside with this class To get this class working the utilities 'plot_tools.py' and 
'image_scale.py' needs to be placed in the same folder (or you can include a
path to the utilities and it will work as well).
"""

# Numpy: 
import numpy as np
from numpy import sum
from numpy import inf, nan, sin, cos, tan, arctan, pi, sqrt, diff, std, diag, argmin, log10, meshgrid
from numpy import mean, median, nanargmax, zeros, ones, ceil, delete, shape, roll, nonzero, std, power
from numpy import arange, array, asarray, size, vstack, hstack, copy, loadtxt, where, savetxt, linspace
from numpy import shape, copy, sort, histogram, meshgrid, ogrid, arctan2, dot
# Packages:
import time, sys, glob 
import pyfits, pylab, scipy, math
# Functions:
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import scipy.ndimage as snd
from scipy.misc import imsave
from astropy.io import fits
from astropy.time import Time
from PIL import Image
# Own functions:
from Plot_Tools import FITS

###########################################################################################################
#                                            DEFINE CLASS                                                 #
###########################################################################################################

class blues(object):
    # INITILIZE THE CLASSE: 
    def __init__(self, path, LF_name, plot, save):
        
        # DEFINE GLOBAL VARIABLES (DGV)

        # Customized informations:
        self.path    = path       # Directory path to data
        self.LF_name = LF_name    # Name of Light Frame (LF)
        self.plot    = plot       # Plot if plot==1
        self.save    = save       # Save if save==1

        # Load light frames:
        self.LF_files = sort(glob.glob('{}{}*'.format(self.path, self.LF_name)))
        self.LF_i = array([pyfits.getdata(str(files)) for files in self.LF_files])  # 
        hdulist = fits.open('{}'.format(str(self.LF_files[0])))   # Get header info
        self.t_exp_LF = hdulist[0].header['EXPTIME']              # Exposure time (CHANGE HEADER NAME!!!) 
        
        # Image dimensions:
        self.n, self.h, self.w = shape(self.LF_i)

        #print self.h, self.w
        #sys.exit()
        # HEADER INFORMATION:
        # load times:
        
        #time[i] = Time(hdulist[0].header['date'], scale='utc').jd  # Scaling to utc time and Julian Date
        #self.DF = 1 # Dark value from header:    
        
###########################################################################################################
#                                            CLASS FUNCTIONS                                              #
###########################################################################################################
            
    def extract_spectra(self, x_stars, y_stars, aperture, background):
        """ 
        This routines perform aperture photometry for stellar images with stars as trails. This function
        returns flux for all the stars that asigned coordinates as well as the Signal to Noise Ratio (SNR)
        for these stars. 
        ----------INPUT:
        x_star         : x stellar coordinate
        y_star         : y stellar coordinate
        aperture       : Aperture to be used: ['ellipse' or 'trace', a, b, q, phi]
        background     : Sky background to be used; 'local' or 'global' 
        """
        print '--------------------------------------------------------------------- aperture_photometry'

        #FITS(self.LF_i[0], 'linear', 2); plt.colorbar(); plt.show() 
        #sys.exit()
        
        #--- CONSTANTS ---#
        gain = 0.73          # Gain of camera: electrons pr ADU (ADU = counts from object)- ajust to camera!
        ron  = 3.3           # Read out noise - ajust to camera!
        con  = 25            # Magnitude constant

        #--- PHOTOMETRY ---#
        # Find fluxes:
        N = len(x_stars) # Number of stellar objects 
        flux_star = zeros((self.n,N))
        SNR_i = zeros((self.n,N))
        for i in range(self.n):  # Loop over all images: if timeseries is available
            for j in range(N):   # Loop over all stars and find flux: using same aperture size
                flux_sky, n_star_pix, flux_star[i][j] = self.aperture(self.LF_i[i], x_stars[j], y_stars[j],\
                                                                      aperture, background)
                SNR_i[i][j] = self.SNR(flux_sky, n_star_pix, flux_star[i][j], gain, ron)

        #--- FINAL CORRECTIONS ---#
        # NORMALIZATION TO OTHER STARS IS LEFT AS A FUTURE PROJECTS:
        print flux_star, flux_sky, SNR_i
        return flux_star, SNR_i
            

                
    def aperture(self, LF, x, y, aperture, background):
        """
        This function calculate the stellar and sky background flux either using a highly elliptical aperture
        or a aperture that traces the the Center Of Flux (COF) to fit the startrails. The routine returns the
        stellar flux 'flux_star', the sky background flux 'flux_sky', and the number of stellar pixels
        'n_pix_star'.
        ----------INPUT:
        LF             : A single Light Frame (LF)
        x, y           : Stellar coordinate
        aperture       : aperture is either: ['ellipse' or 'trace', a, b, q, phi]
        background     : sky background is either: 'local' or 'global'
        """
        # Aperture only handle integers:
        x, y = int(x), int(y)

        # Take time
        start_time = time.time()
        
        # Grid for aperture:
        x_grid, y_grid = meshgrid(range(self.w), range(self.h))
        grid = LF[y_grid, x_grid]  # Grid
        xx = x_grid - x            # Displacement for stellar x coor
        yy = y_grid - y            # Displacement for stellar y coor
        
        # Ellipse parameters:
        a   = aperture[1]                # Semiminor/lenght axis
        b   = aperture[2]                # Semimajor/width axis 
        q   = aperture[3]                # Width of background band
        phi = math.radians(aperture[4])  # Tilt angle: [0:180] deg
        
        # Axis of background: 
        a_sky = a + q   # Semiminor/lenght axis
        b_sky = b + q   # Semimajor/width axis

        # # Global background:
        # if background=='global':
        #     flux_sky = self.global_sky_background(LF)
        
        #--- ELLIPTIC APERTURE ---#
        if aperture[0]=='ellipse':

            # Parametrasation and rotation of the ellipse (1 = x^2/a^2 + y^2/b^2)
            phi = phi - pi/2 # The ellipse have an offset from unity circle 
            EE_star  = (xx*cos(phi)+yy*sin(phi))**2/a**2     + (xx*sin(phi)-yy*cos(phi))**2/b**2
            EE_sky   = (xx*cos(phi)+yy*sin(phi))**2/a_sky**2 + (xx*sin(phi)-yy*cos(phi))**2/b_sky**2
            
            # Local background:
            if background=='local':
                E_sky    = ((EE_star>1)*(EE_sky<1))*grid   # Sky background determined by width q 
                sky      = E_sky[nonzero(E_sky)]           # Sky pixel values
                flux_sky = 3*median(sky) - 2*mean(sky)     # Robust sky background flux
            
            # Star: 
            E_star     = (EE_star<=1)*grid
            star       = E_star[nonzero(E_star)]-flux_sky  # Stellar corrected pixels
            n_pix_star = sum(EE_star<=1)                   # Number of used star pixels
            flux_star  = sum(star)                         # Flux from star
            
        #--- TRACE APERTURE ---#
        if aperture[0]=='trace':

            # Star initial:
            CC_star  = sqrt(xx**2 + yy**2) - a      # Start frame used to trace from:
            star_img = (CC_star<=1)*grid            # Star images
            star_val = star_img[nonzero(star_img)]  # Stellar pixels
            n_pix_star = len(star_val)              # Number of pixels in circle
                
            # Loop-step in x or y depends on phi:
            # Step in x and finds y centroid for each step:
            if 0<=phi<=pi/4 or pi*3/4<=phi<=pi:
                step = 'x step'
                x_step = range(b)
                y_step = zeros(b)
            # Step in y and finds x centroid for each step:
            if pi/4<phi<pi*3/4:
                step = 'y step'
                x_step = zeros(b)
                y_step = range(b)

            # Loop over trace step:
            x_cen = zeros(b)
            y_cen = zeros(b)
            #star_true = zeros((b, self.h, self.w))
            #sky_true  = zeros((b, self.h, self.w))
            for i in range(b):
                # Find Center Of Flux (COF):
                y_cen[i], x_cen[i] = self.center_of_flux(star_img, n_pix_star)
                if step=='x step': XX = x + x_step[i]; YY = y_cen[i] + y_step[i]
                if step=='y step': YY = y + y_step[i]; XX = x_cen[i] + x_step[i]
                # A circular aperture is used to trace with:
                CC_star = sqrt((x_grid-XX)**2 + (y_grid-YY)**2) - a
                star_true  = (CC_star<=1)                      # Array of true and false statement: star
                star_img   = star_true*grid                 # Image: star
                n_pix_star = len(star_img[nonzero(star_img)])  # Number of pixels: star
                # If local background:
                # if background=='local':
                #     CC_sky  = sqrt((x_grid-XX)**2 + (y_grid-YY)**2) - a_sky
                #     sky_true[i] = ((CC_star>1)*(CC_sky<1))*grid  # Array of true and false statement: sky
                # Take time of routine:
                if i==round(b*0.25):
                    print ('Done 25  procent --- %0.5s seconds ---' % (time.time() - start_time))       
                if i==round(b*0.50):
                    print ('Done 50  procent --- %0.5s seconds ---' % (time.time() - start_time))
                if i==round(b*0.75):
                    print ('Done 75  procent --- %0.5s seconds ---' % (time.time() - start_time))
                if i==round(b-1):
                    print ('Done 100 procent --- %0.5s seconds ---' % (time.time() - start_time))

            # Local sky flux:
            # if background=='local':
            #     star_false = np.logical_not(np.sum(star_true, axis=0))  # Invert to get rid of star
            #     sky_true_x = np.sum(sky_true, axis=0) > 0
            #     sky_img  = sky_true_x.astype(np.int)*star_false*LF      # Here bol: True*False = False
            #     sky      = sky_img[nonzero(sky_img)]
            #     flux_sky = 3*median(sky) - 2*mean(sky)                  # Robust sky flux
            
            # # Stellar flux: 
            # star_true_x = np.sum(star_true, axis=0) > 0         # Bool array of stellar pixels
            # star_img    = star_true_x.astype(np.int)*LF         # Stellar image
            # star        = star_img[nonzero(star_img)]-flux_sky  # Stellar corrected pixels
            # n_star_pix  = len(star)
            # flux_star   = sum(star_img)
            
        # PLOT IF YOU LIKE::
        if self.plot==1:
            # Elliptic aperture with local background:
            if aperture[0]=='ellipse':
                from plot_tools import plot_ellipse
                FITS(LF, 'linear', 2)
                plot_ellipse(a,     b,     math.degrees(phi), x, y, 'g')  # Stellar aperture
                plot_ellipse(a_sky, b_sky, math.degrees(phi), x, y, 'm')  # Background aperture
                plt.show()
            # Box aperture with local background:
            if aperture[0]=='trace':
                 t = linspace(0, 2*pi)
                 FITS(LF, 'linear', 2)
                 [plt.plot(a_sky*cos(t)+(x+x_step[i]), a_sky*sin(t)+y_cen[i], 'b-') for i in range(b)]
                 [plt.plot(a*cos(t)+(x+x_step[i]), a*sin(t)+y_cen[i], 'g-') for i in range(b)]
                 plt.show()

        sys.exit()
        return flux_sky, n_pix_star, flux_star

    
    
    
    def center_of_flux(self, LF, n_pix):
        """
        This function finds the center of flux for all desired stellar object. Here LF is the masked image
        thus every pixel is set to zero except the star) and n_pix is the number of pixels one wish to use
        in order to find the COF.
        """
        # Loops over all pixels:
        LF_copy  = copy(LF)     # Copy to avoid overwriting
        
        flux_max = zeros(n_pix)
        x_max = zeros(n_pix)
        y_max = zeros(n_pix)
        pixel = zeros(n_pix)
        for j in range(n_pix):
            flux_max[j] = np.max(LF_copy)               # Maximum value for array
            max_dex = np.where(LF_copy == flux_max[j])  # Find row, column for min value
            x_max[j] = max_dex[0][0]                    # max for x coordinate
            y_max[j] = max_dex[1][0]                    # max for y coordinate
            pixel[j] = j
            # Min pixel is et to max in order to find the next min:
            LF_copy[int(y_max[j]), int(x_max[j])] = 0

        # Flux center is found:
        flux = sum(pixel)
        cen_x = 1/flux*dot(pixel,x_max)              # Flux-center in x
        cen_y = 1/flux*dot(pixel,y_max)              # Flux-center in y
        return cen_x, cen_y
