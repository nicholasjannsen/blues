""" 
UTILITY DESCRIBTION:
--------------------
As the name states this is a utility with various plot tools to make it easier to make nice plots. This 
utility has the following overview:
"""

# Numpy:
import numpy as np

# Python functions: 
import time, sys, pylab, math, julian

# Matplotlib:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc, cm, gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# Axes tools:
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable

###########################################################################################################
#                                            GENERAL PLOTS                                                #
###########################################################################################################

def PLOT(data, mark, xlab, ylab, title=None, subplot=0, legend=1,  axis=[1,1]):
    """
    General function to make fast plots in one command line:
    --------INPUT:
    data       (array)  : Data structure e.g. [data0, data1];  data0 and data1 have a x, y coloumn.
    mark       (list)   : If one have 2 datasets use e.g. ['b-', 'k.']
    xlab, ylab (string) : Labels on x and y
    title      (string) : Title
    legpos     (float)  : This can be 1, 2, 3, and 4 corresponding to each quadrant.
    subplot    (float)  : Different types of subplots.
    axis       (list)   : Procentage edge-space in x and y. E.g. [1, 5] to 1% in x and 5% in y. 
    """
    # Type of subplot:
    if subplot is not 0: plot_subplot(subplot)
    # Plot data:
    if legend is 1:
        for i in range(len(data)):
            plt.plot(data[i][:,0], data[i][:,1], mark[i])
        plot_settings(xlab, ylab, title)     
    if legend is not 1:
        for i in range(len(data)):
            plt.plot(data[i][:,0], data[i][:,1], mark[i], label=legend[i+1])
        plot_settings(xlab, ylab, title, legend[0])
    # Axes setting:
    plot_axis(data[0][:,0], data[0][:,1], axis[0], axis[1])
    plt.show()


def SURF(x, y, z, xlab, ylab, zlab, title=None):
    # Find (x, y) value for maximum peak:
    z_max   = np.max(z)
    z_max_i = np.where(z==z_max)
    print('Best Period: {:.6f} days'.format(x[z_max_i[0][0]]))
    print('Best Phase : {:.6f} days'.format(y[z_max_i[1][0]]))
    # 3D plot:
    y, x = np.meshgrid(y, x)
    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='bwr', linewidth=20, antialiased=False)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # cm.coolwarm
    # Axes labels and title:
    ax.set_xlabel(xlab, fontsize=13); ax.tick_params(axis='x', labelsize=13)
    ax.set_ylabel(ylab, fontsize=13); ax.tick_params(axis='y', labelsize=13)
    ax.set_zlabel(zlab, fontsize=13); ax.tick_params(axis='z', labelsize=13)
    if title is not None: plt.title(title,    fontsize=15)
    # Extra settings:
    # ax.invert_xaxis()                          # Invert x-axis
    # ax.view_init(30, 45)                       # Viewing angle 
    # fig.colorbar(surf, shrink=0.5, aspect=8)   # Colorbar
    plt.show()
 
def HIST(hist, bins, xlab, ylab, title=None):
    plt.hist(hist, bins, edgecolor='k', alpha=1, log=True)
    #plot_settings(xlab, ylab, title)
    

def linear(img, sigma=2, img_min=None, img_max=None):
    """ Performs linear scaling of the input np array. """
    img_min, img_max = img.mean()-sigma*img.std(), img.mean()+sigma*img.std()
    imageData=np.array(img, copy=True)
    #if scale_min == None: scale_min = imageData.min()
    #if scale_max == None: scale_max = imageData.max()
    imageData = imageData.clip(min=img_min, max=img_max)
    imageData = (imageData -img_min) / (img_max - img_min)
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    indices = np.where(imageData > 1)
    imageData[indices] = 1.0
    return imageData


def FITS(img, sigma=2, xlab=None, ylab=None, colorbar=None):
    """ Easy plot of pixel array data """
    plt.figure()
    plt.imshow(linear(img, sigma), cmap='Blues', origin='lower')
    # Labels:
    if xlab is None and ylab is None: xlab, ylab = r'$x$ (pixel)', r'$y$ (pixel)'
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

    
def FITS_colorbar_multi(img, xlab, ylab, ds=None):
    # Which scale that should be used:
    from Plot_Tools import linear
    if scale=='linear': scale=linear
    if scale=='sqrt'  : scale=sqrt
    if scale=='log'   : scale=log
    if scale=='asinh' : scale=asinh
    if ds==None: ds=2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(nrows=1, ncols=len(img))
    for ax in axes.flat():
        img_min, img_max = img[i].mean()-ds*img[i].std(), img[i].mean()+ds*img[i].std()
        # Prepare for scaled colorbar:
        im = ax.imshow(scale(img[i], img_min, img_max), cmap='Blues', origin='lower')     
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax  = divider.append_axes('right', size='5%', pad=0.05) # right, left, top, bottom    
        # Make colorbar and append ylabel and axis labels:
        cbar = plt.colorbar(im, cax=cax)# orientation='horizontal')
    #cbar.ax.set_ylabel('Normalized Counts')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
###########################################################################################################
#                                          BlueSONG PIPELINES                                             #
###########################################################################################################

def plot_image_reduction(BF, DF, FF, TA, SF):
    print('Bias -- std: {:.5g}, mean: {:.5g}'.format(np.std(BF), np.mean(BF)))
    print('Dark -- std: {:.5g}, mean: {:.5g}'.format(np.std(DF), np.mean(DF)))
    print('Flat -- std: {:.5g}, mean: {:.5g}'.format(np.std(FF), np.mean(FF)))
    # Plot calibration images:
    xlab = 'Dispersion (pixels)'; ylab = 'Cross Dispersion (pixels)'
    FITS(BF); FITS(DF); FITS(FF); FITS(TA); FITS(SF)

#-----------------------------------------------------------------------------------------------------------
    
def plot_find_ref_cen_pos(center_rows_median, center_row_median_convolved, len_cross, smooth_win, ref_cen_pos):
    """ This plot shows how the central reference point are found in each order """
    # Add offset to cross-axis and 0.025 in disp-axis to colvolved:
    len_cross = len(center_rows_median)
    add_cross = np.ones(len_cross)*int(smooth_win/2)
    data1 = np.vstack([center_rows_median/(max(center_rows_median)), range(len_cross)]).T
    data2 = np.vstack([center_row_median_convolved/(max(center_row_median_convolved)) + 0.01, \
                       range(len_cross) - add_cross]).T
    # Plot figure:
    plt.figure()
    plt.plot(data1[:,0], data1[:,1],     'b-',  label='Collapsed: Median-filter', alpha=0.5)
    plt.plot(data2[10:,0], data2[10:,1], 'g--', label='Convolved: Sum-filter')
    for i in range(len(ref_cen_pos)): plt.plot([0, 1.03], [ref_cen_pos[i], ref_cen_pos[i]], 'k:')
    plt.plot([0, 1], [ref_cen_pos[0], ref_cen_pos[0]], 'k:', label='Ref Orders center')
    plt.xlabel('Normalized Counts'); plt.ylabel('Cross Dispersion (pixels)')
    plt.legend(loc='lower right')
    plt.show()

def plot_trace_order(ridge_pos_disp, ridge_pos_cross, order_trace, order_traced, \
                     order_trace_disp, cen_disp, ref_cen_pos):
    """ This plot shows how the order are traced and fitted using sigma-clipping and a polynomial fit """
    plt.figure()
    plt.plot(ridge_pos_disp, ridge_pos_cross, 'b.', alpha=0.2)
    for i in range(4): plt.plot(order_trace['order_{}'.format(i)][0],\
                                order_trace['order_{}'.format(i)][1], 'k.')
    for i in range(4): plt.plot(order_trace_disp, np.polyval(order_traced['order_{}'.format(i)],\
                                order_trace_disp), 'c-', linewidth='1.5')
    # Appearence of labels:
    plt.plot(ridge_pos_disp[0], ridge_pos_cross[0], 'b.', alpha=0.3, label='Traced ridge')
    plt.plot(order_trace['order_0'][0], order_trace['order_0'][1], 'k.', label='Final ridge')
    plt.plot(order_trace_disp, np.polyval(order_traced['order_{}'.format(0)],\
                                          order_trace_disp), 'c-', linewidth='1.5', label='Polynomial fit')
    plt.plot(ref_cen_pos*0 + cen_disp, ref_cen_pos, 'r*', label='Ref posisions', markersize='7')
    plt.xlabel('Dispersion (pixels)'); plt.ylabel('Cross Dispersion (pixels)')
    plt.legend(loc='lower right', ncol=2)
    plt.ylim(0, 450)
    plt.show()

#-----------------------------------------------------------------------------------------------------------

def plot_optimal_width(widths, order, blaze_max, index_max, flux_inter, snr, optimal_order_width):
    # Find best FWHM of Gauss fit to highest S/N ratio width:
    from lmfit.models import GaussianModel
    gmodel = GaussianModel()
    # Make Gauss fit:
    x       = np.arange(len(order[index_max]))
    profile = order[index_max] - flux_inter - np.min(order)/2
    result  = gmodel.fit(profile, x=x, amplitude=blaze_max, center=optimal_order_width)
    fit     = result.best_fit
    fwhm    = result.params['fwhm'].value
    print(result.fit_report())
    print(optimal_order_width)
    # PLOT:
    fig, ax = plt.subplots(1, 2, sharex=True)
    # Plot S/N ratio:
    ax1 = plt.subplot(121)
    ax1.plot(widths[0:-1-1], snr[0:-1-1], 'mo--')
    ax1.axvline(optimal_order_width, color='b', linestyle='--')
    ax1.set_xlabel('Spatial width (pixels)')
    ax1.set_ylabel('S/N ratio')
    # Plot gauss fit:
    xoff = (x[-1]-x[0])/2
    ax2 = plt.subplot(122)
    ax2.plot(x-xoff, profile/max(profile), 'r+')
    ax2.plot(x-xoff, fit/max(profile), 'k-', label='Gauss fit')
    #--------
    ax2.axvline(x=(np.nanargmax(fit)-fwhm/2-xoff), color='orange', linestyle='--', label='FWHM')
    ax2.axvline(x=(np.nanargmax(fit)+fwhm/2-xoff), color='orange', linestyle='--')
    #--------
    ax2.axvline(x=(np.nanargmax(fit)-optimal_order_width/2-xoff), color='b', linestyle='--', \
                label=r'(S/N)$_{\text{max}}$')
    ax2.axvline(x=(np.nanargmax(fit)+optimal_order_width/2-xoff), color='b', linestyle='--')
    #--------
    ax2.axvline(x=(np.nanargmax(fit)-optimal_order_width/1.5-xoff), color='g', linestyle='--', \
                label=r'$4/3 \times \text{(S/N)}_{\text{max}}$')
    ax2.axvline(x=(np.nanargmax(fit)+optimal_order_width/1.5-xoff), color='g', linestyle='--')
    #--------
    ax2.axvline(x=(np.nanargmax(fit)-(optimal_order_width-3)/2-xoff), color='r', linestyle='--', \
                label=r'Final width')
    ax2.axvline(x=(np.nanargmax(fit)+(optimal_order_width-3)/2-xoff), color='r', linestyle='--')
    #--------
    ax2.set_xlabel('Central width (pixels)')
    ax2.set_ylabel('Norm. counts')
    # Extra settings:
    ax2.legend(loc='upper left', fontsize=11, ncol=5, bbox_to_anchor=(-1.2, 1.15))
    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    plt.show()
    
def plot_inter_order_mask(data, inter_order_mask):
    mask = data.T*inter_order_mask.T
    # Plot:
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(linear(data.T), cmap='Blues', origin='lower', alpha=1.0)
    im = ax.imshow(linear(mask),   cmap='Blues', origin='lower', alpha=0.73)
    # Settings:
    ax.set_xlabel('Dispersion (pixels)')
    ax.set_ylabel('Cross (pixels)')
    # Colorbar:
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='1.5%', pad=0.0)
    cbar    = plt.colorbar(im, cax=cax)
    plt.ylabel('Norm. Counts')
    plt.show()

def plot_background_fits(s_disp, s_cros, poly_order_y, poly_order_x, \
                         yy, yi0, yi1, yi2, ym0, ym1, ym2, yfit0, yfit1, yfit2, \
                         xx, xi0, xi1, xi2, xm0, xm1, xm2, xfit0, xfit1, xfit2):
    font1, font2 = 12.5, 15
    # Sizes for flat: 
    m1, m2, m3, m4, m5, m6 = 5, 3, 3, 6, 4, 4
    a1, a2, a3, a4, a5, a6 = 0.2, 0.5, 0.5, 0.01, 0.05, 0.1
    # Sizes for star:
    m1, m2, m3, m4, m5, m6 = 5, 3, 3, 4, 3, 3
    a1, a2, a3, a4, a5, a6 = 0.4, 0.5, 0.5, 0.1, 0.1, 0.2
    #---------------------
    grid = plt.GridSpec(7, 4, wspace=0.0, hspace=0.0)
    #---------------------
    ax1 = plt.subplot(grid[:6,0])
    ax1.plot(yi0, ym0, 'b.', alpha=a1, markersize=m1)#, label='Row {}'.format(s_disp[0])
    ax1.plot(yi1, ym1, 'g^', alpha=a2, markersize=m2)#, label='Row {}'.format(s_disp[1])
    ax1.plot(yi2, ym2, 'rd', alpha=a3, markersize=m3)#, label='Row {}'.format(s_disp[2])
    ax1.plot(yy, yfit0, 'k-',  linewidth=1.5, label='{}. order polyfit'.format(poly_order_y))
    ax1.plot(yy, yfit1, 'k--', linewidth=1.5)#, label='{}. order polyfit'.format(poly_order_y))
    ax1.plot(yy, yfit2, 'k:',  linewidth=1.5)#, label='{}. order polyfit'.format(poly_order_y))
    #---------------------
    ax2 = plt.subplot(grid[:6,1])
    ax2.plot(xi0, xm0, 'b.', alpha=a4, markersize=m4) # Not used for the labels
    ax2.plot(xi1, xm1, 'g^', alpha=a5, markersize=m5)
    ax2.plot(xi2, xm2, 'rd', alpha=a6, markersize=m6)
    ax2.plot(xx, xfit0, 'k-',  linewidth=1.5, label='{}. order polyfit'.format(poly_order_x))
    ax2.plot(xx, xfit1, 'k--', linewidth=1.5)#, label='{}. order polyfit'.format(poly_order_x))
    ax2.plot(xx, xfit2, 'k:',  linewidth=1.5)#, label='{}. order polyfit'.format(poly_order_x))
    # Settings:
    ax1.legend(loc='upper left', fontsize=font1, ncol=1, bbox_to_anchor=(-0.035, 1.15))
    ax2.legend(loc='upper left', fontsize=font1, ncol=1, bbox_to_anchor=(-0.035, 1.15))
    ax1.set_xlabel('Cross (pixels)',           fontsize=font2)
    ax2.set_xlabel('')
    ax2.set_xlabel('Dispersion (pixels)',      fontsize=font2)
    ax1.set_ylabel('Counts',                   fontsize=font2)
    ax1.tick_params(axis='both', which='both', labelsize=font2)
    ax2.tick_params(axis='both', which='both', labelsize=font2)
    ax2.set_yticklabels([''])
    ax1.set_xticklabels(['0', '0', '100', '', ''])
    ymin1, ymax1 = min(min(ym0), min(ym1), min(ym2)), max(max(ym0), max(ym1), max(ym2))
    ymin2, ymax2 = min(min(xm0), min(xm1), min(xm2)), max(max(xm0), max(xm1), max(xm2))
    ax1.set_ylim(ymin1-ymin1*0.05, ymax1+ymax1*0.05)
    ax2.set_ylim(ymin2-ymin2*0.05, ymax2+ymax2*0.05)
    # Extra:
    # ax2.plot(xi0[0], xm0[0], 'b.', alpha=0.4, markersize=mark2)#, label='Column {}'.format(s_cros[0])
    # ax2.plot(xi1[0], xm1[0], 'g^', alpha=0.4, markersize=mark1)#, label='Column {}'.format(s_cros[1])
    # ax2.plot(xi2[0], xm2[0], 'rd', alpha=0.4, markersize=mark1)#, label='Column {}'.format(s_cros[2])
    plt.show()
    
def plot_background(data):
    # Plot:
    fig, ax = plt.subplots()
    im = ax.imshow(linear(data.T), cmap='Blues', origin='lower')
    # Settings:
    ax.set_xlabel('Dispersion (pixels)')
    ax.set_xticklabels([])
    ax.set_ylabel(r'Cross Dispersion (pixels)')
    # Colorbar:
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3%', pad=0.0)
    cbar    = plt.colorbar(im, cax=cax)
    plt.ylabel(r'Counts')
    plt.show()


def plot_background_residuals(F_back, S_back, S):
    F = F_back/np.max(F_back)
    S = S_back/np.max(S_back)
    I = F - S
    fig, ax = plt.subplots(2,1)
    # Plot:
    im0 = ax[0].imshow(linear(S.T), cmap='Blues', origin='lower')
    im1 = ax[1].imshow(I.T, vmin=np.min(I), vmax=np.max(I), cmap='Blues', origin='lower')     
    # Settings:
    fig.subplots_adjust(hspace=-0.50)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels(['', '', '200', '400'])
    # Colorbar ax0:
    cbar0 = fig.add_axes([0.9, 0.510, 0.015, 0.227])
    cbar1 = fig.add_axes([0.9, 0.253, 0.015, 0.228])
    fig.colorbar(im0, cax=cbar0)
    fig.colorbar(im1, cax=cbar1)
    # Labels:
    ax[0].annotate('(a)', xy=(50,280), fontsize=15)
    ax[1].annotate('(b)', xy=(50,280), fontsize=15)
    ax[1].set_xlabel('Dispersion (pixels)')
    ax[0].set_ylabel(r'Cross Dispersion (pixels)\qquad\qquad\qquad\tiny.')
    cbar0.set_ylabel('Norm. Counts')
    cbar1.set_ylabel('Residuals')
    plt.show()
    
#-----------------------------------------------------------------------------------------------------------

def plot_optimal_extraction(img):
    # Prepare for scaled colorbar:
    gs = gridspec.GridSpec(1, 3)
    #------------
    # 1. SUBPLOT:
    #------------
    ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)
    # Plot pixel image:
    #im = FITS(img.T, ax=ax1)
    im = ax1.imshow(linear(img.T), cmap='Blues', origin='lower')
    # Plot red and green boxes (given as cornerpoint and height and width):
    ax1.add_patch(Rectangle((7.5,  -0.4), 1, len(img[:,0])-0.3, fill=None, edgecolor='r', linewidth=1.2))
    ax1.add_patch(Rectangle((29.5, -0.4), 1, len(img[:,0])-0.3, fill=None, edgecolor='g', linewidth=1.2))
    # Labels:
    ax1.set_xlabel(r'$\lambda$ (pixel)')
    ax1.set_ylabel(r'$x$ (pixel)')
    #------------
    # 2. SUBPLOT:
    #------------
    ax2 = plt.subplot2grid((9,3), (0,2), colspan=1, rowspan=5)
    # Cross cut in the image:
    l1 = img[:,8]; l2 = img[:,10]#+100*np.ones(len(l1))
    # Plot step plots:
    ax2.step(l1/np.max(img), range(len(l1)), color='r')
    ax2.step(l2/np.max(img), range(len(l2)), color='g')
    # Remove ticks and numbers from both axes:
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # Plot equation inside plot:
    ax2.text(-0.05, 6.5, r'$\sum\limits_x \text{P}_{x\lambda}=1$', fontsize=15)
    # Put colorbar from first plot as the x label (as they are shared): 
    divider = make_axes_locatable(ax2)
    cax  = divider.append_axes('bottom', size='10%', pad=0.0) # right, left, top, bottom    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('Normalized Counts')
    # Plot figure with small ajustment:
    plt.subplots_adjust(wspace=0, hspace=-0.03)
    plt.show()
    
#-----------------------------------------------------------------------------------------------------------
    
def plot_arc_peak(s_neg, s_conv, peaks_limit_dex, peaks_limit_val, \
                     peaks_all_dex, peaks_all_val, peak, limits):
    fig, ax = plt.subplots()
    # MAKE PLOTS:
    ax.plot(s_neg,                        '-', color='grey',   label='Normal spectrum')
    ax.plot(peaks_all_dex, peaks_all_val, 'x', color='orange', label='Normal peaks')
    ax.plot(s_conv,                       'k', color='k'   ,   label='Convolved spectrum')
    ax.axvline(limits[0], color='deeppink', linestyle='-.')
    ax.axvline(limits[1], color='deeppink', linestyle='-.')
    ax.plot(peaks_limit_dex, peaks_limit_val, '^', color='deeppink', markeredgecolor='k', markersize=7, \
             label='Smooth peaks')
    ax.axvline(x=peak, color='b', linestyle='--', label='Line center reference')
    # AXES SETTINGS:
    ax.set_xlabel(r'$\lambda$ (pixel)')
    ax.set_ylabel('Norm. counts')
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1], handles[3], handles[4]]
    labels  = [labels[0], labels[2], labels[1], labels[3], labels[4]]
    ax.legend(handles, labels, loc='best')
    ax.set_xlim((0, len(s_neg)))
    hk_median = np.median(s_neg)
    ax.set_ylim((hk_median-1.5*abs(hk_median), hk_median+2.5*abs(hk_median)))
    plt.show()       

def plot_arc_fit(l_obs, l_teo, coefs, poly_order, residuals, chi2r, sigma, text):
    # Calculate fit parameters:
    p         = np.poly1d(coefs)
    xp        = np.linspace(min(l_obs), max(l_obs), 1e3)
    # Plot fit:
    fig = plt.figure()
    ax0 = plt.subplot2grid((4,1), (0,0), rowspan=3)
    ax0.plot(xp,    p(xp), 'r-', label='{}. order polyfit'.format(poly_order))
    ax0.plot(l_obs, l_teo, 'k+', label='ThAr lines')
    # Plot residuals:
    ax1 = plt.subplot(414)    
    ax1.plot(l_obs, residuals,         'k+')
    ax1.plot(xp,    np.zeros(len(xp)), 'r--')
    # Settings:
    xanopos = max(xp)-(max(xp)-min(xp))*0.2
    yanopos = (max(l_obs)-min(l_obs))
    ax0.annotate(r'$\chi^2_r$ = {:.4f}'.format(chi2r[0]), (xanopos, max(l_teo)-yanopos*0.8))
    ax0.annotate(r'$\sigma$ = {:.4f}'.format(sigma),      (xanopos, max(l_teo)-yanopos*0.9))
    ax0.set_title(text)
    ax0.set_xlabel('')
    ax0.set_xticklabels('')
    ax0.set_ylabel(r'$\lambda_{\text{atlas}}$ (Å)')
    ax1.set_xlabel(r'$\lambda_{\text{obs}}$ (Å)')
    ax1.set_ylabel('Residuals (Å)')
    #ax1.set_ylim([-0.025, 0.025])
    fig.subplots_adjust(hspace=0)
    ax0.legend(loc='best')
    plt.show()
    
def plot_arc_scale(l_obs0, l_teo0, l_obs1, l_teo1, l, obs_results):
    """FIGURE MADE TO SPANS ENTIRE SCREEN"""
    # Unpack results from peak_finder:
    text, img, COF, radius = obs_results[0], obs_results[1], obs_results[2], obs_results[3]
    #----------
    # SUBPLOTS:
    #----------
    font = 17
    fig  = plt.figure()
    # 1. subplot:
    ax1 = fig.add_subplot(4,1,(1,3))
    for i in range(len(l_obs0)): ax1.axvline(x=l_obs0[i], ymax=0.85, color='r', linestyle='--', alpha=1)
    for i in range(len(l_obs1)): ax1.axvline(x=l_obs1[i], ymax=0.85, color='r', linestyle='-',  alpha=1)
    for i in range(len(l_teo0)): ax1.axvline(x=l_teo0[i], ymax=0.85, color='b', linestyle='--', alpha=1)
    for i in range(len(l_teo1)): ax1.axvline(x=l_teo1[i], ymax=0.85, color='b', linestyle='-',  alpha=1)
    ax1.plot(l, img.sum(axis=1)/np.max(img.sum(axis=1)), 'k-', label='ThAr spectrum')
    # 2. subplot:
    ax2 = fig.add_subplot(313)
    ax2.imshow(linear(img.T), cmap='Blues')
    ax2.scatter(COF[:,0], COF[:,1], s=radius*12, facecolors='none', edgecolors='r', marker='s')
    # Annotation:
    for i in range(len(l_teo0)):
        ax1.annotate(l_teo0[i], (l_teo0[i]-0.5, 1.15), rotation=45, fontsize=11, color='b')
    # Labels:
    ax1.set_ylabel('Normalized Counts',  fontsize=font)
    ax1.set_xlabel(r'$\lambda$ (Å)', fontsize=font)
    ax1.tick_params(labelsize=font)
    #------
    #ax1.set_xticklabels([])
    #ax2.set_yticklabels(['0', ''])
    ax2.set_xlabel(r'$\lambda$ (pixel)', fontsize=font)
    ax2.set_ylabel(r'$x$ (pixel)', fontsize=font)
    ax2.tick_params(labelsize=font)
    # Axes:
    ax1.set_xlim((min(l), max(l)))
    ax1.set_ylim((0, 1.2))
    ax2.set_xlim((0, 2749))
    ax2.invert_yaxis()
    plt.show()

def plot_arc_check(l, img, l_ca, text):
    fig  = plt.figure()
    # Plot:
    ax1 = fig.add_subplot(4,1,(1,3))
    for i in range(len(l_ca)): ax1.axvline(x=l_ca[i], ymax=1, color='g', linestyle='-.')
    ax1.plot(l[0], img[0].sum(axis=1)/np.max(img[0].sum(axis=1)), 'b-')
    ax1.plot(l[1], img[1].sum(axis=1)/np.max(img[1].sum(axis=1)), 'k-')
    # Settings:
    plt.title(text)
    plt.xlabel(r'$\lambda_{\text{obs}}$ (Å)')
    plt.ylabel(r'Counts')
    plt.show()

def plot_arc_illustration(l_obs0, l_teo0, l_obs1, l_teo1, l, obs_results):
    """FIGURE MADE TO SPANS ENTIRE SCREEN"""
    # Unpack results from peak_finder:
    text, img, COF, radius = obs_results[0], obs_results[1], obs_results[2], obs_results[3]
    #----------
    # SUBPLOTS:
    #----------
    font = 12
    fig  = plt.figure()
    h = 0.85
    # 1. subplot:
    ax1 = fig.add_subplot(10,1,(1,9))
    for i in range(len(l_obs0)): ax1.axvline(x=l_obs0[i], ymax=h, color='r', linestyle=':', alpha=1)
    #for i in range(len(l_obs1)): ax1.axvline(x=l_obs1[i], ymax=0.85, color='r', linestyle='-',  alpha=1)
    for i in range(len(l_teo0)): ax1.axvline(x=l_teo0[i], ymax=h, color='b', linestyle=':', alpha=1)
    for i in range(len(l_teo1)): ax1.axvline(x=l_teo1[i], ymax=h, color='limegreen', linestyle='-', alpha=1)
    ax1.plot(l, img.sum(axis=1)/np.max(img.sum(axis=1)), 'k-', label='ThAr spectrum')
    # 2. subplot:
    ax2 = fig.add_subplot(10,1,10)
    ax2.imshow(linear(img.T), cmap='Blues')
    ax2.scatter(COF[:,0], COF[:,1], s=radius*12, facecolors='none', edgecolors='r', marker='s')
    #-----------------
    # GLOBAL SETTINGS:
    #-----------------
    # Legend:
    ax1.axvline(x=l_obs0[0], ymax=h, color='r',         linestyle='--', alpha=0.4, label='COF')
    ax1.axvline(x=l_teo0[i], ymax=h, color='b',         linestyle=':',  alpha=1.0, label='Atlas lines')
    ax1.axvline(x=l_teo1[i], ymax=h, color='limegreen', linestyle='-',  alpha=0.2, label='Final lines')
    ax1.legend(bbox_to_anchor=(0.93, 1.14), ncol=4, fontsize=font)
    # Annotation:
    for i in range(len(l_teo0)):
        ax1.annotate(l_teo0[i], (l_teo0[i]-0.5, 1.15), rotation=45, fontsize=8, color='darkblue')
    # Labels:
    ax1.set_ylabel('Norm. Counts',  fontsize=font)
    ax1.tick_params(labelsize=font)
    ax1.set_xticklabels([])
    ax1.set_yticklabels(['', '', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    #ax2.set_yticklabels(['0', ''])
    ax2.set_xlabel(r'$\lambda$ (pixel)', fontsize=font)
    ax2.set_ylabel(r'$\Delta x$', fontsize=font)
    ax2.tick_params(labelsize=font)
    # Axes:
    ax1.set_xlim((l[820], l[1850]))
    ax2.set_xlim((820, 1850))
    ax1.set_ylim((-0.02, 1.2))
    ax2.invert_yaxis()
    fig.subplots_adjust(hspace=-0.35)
    plt.show()

#-----------------------------------------------------------------------------------------------------------

def plot_blaze(s_orders, f_orders, f_lincor, dif_max):
    # Normalize everyting:
    yoff = 2500
    dy = np.max(f_lincor[1]/dif_max[1]) + yoff
    # Blaze + un-blaze spectrum:
    plt.figure()
    plt.plot(s_orders[1]+yoff, 'k-', linewidth=0.5, label='Object with cosmics')
    plt.plot(s_orders[0],      'k-', linewidth=0.5)
    plt.plot(f_orders[1]/dif_max[1]+yoff, 'r-', label='Blaze hot pixels')
    plt.plot(f_orders[0]/dif_max[0],      'r-')
    plt.plot(f_lincor[1]/dif_max[1]+yoff, 'b-', label='Blaze Ca order')
    plt.plot(f_lincor[0]/dif_max[0],      'c-', label='Blaze order below')
    plt.legend(loc='best')
    plt.xlabel(r'$\lambda$ (pixel)')
    plt.ylabel('Counts')
    plt.ylim(-dy*0.1, dy+dy*0.1)
    plt.show()

def plot_deblaze(s_blazecor):
    # Blaze + un-blaze spectrum:
    plt.figure()
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    # Order #57:
    ax1 = plt.subplot(grid[0,:])
    ax1.plot(s_blazecor[1]/np.median(s_blazecor[1]), 'k-')
    ax1.axhline(1, color='b', linestyle='--')
    ax1.set_ylim(-0.2, 3)
    # Order #58:
    ax2 = plt.subplot(grid[1,:])
    ax2.plot(s_blazecor[0]/np.median(s_blazecor[0]), 'k-')
    ax2.axhline(1, color='b', linestyle='--')
    ax2.set_ylim(-0.2, 3)
    # Settings:
    ax2.set_xlabel(r'$\lambda$ (pixel)')
    ax2.set_ylabel('Normalized Counts', y = 1)
    plt.show()

#-----------------------------------------------------------------------------------------------------------

def plot_merge(s, l, l_ca):
    plt.figure()
    plt.plot(l, s, '-', color='k', linewidth=0.5)
    for i in range(len(l_ca)):
        plt.axvline(x=l_ca[i],   ymax=1, color='g',    linestyle='-.', alpha=1.0)
    plt.axhline(0, color='r', linestyle='--')
    # Settings:
    plt.xlabel(r'$\lambda_{\text{obs}}$ (Å)')
    plt.ylabel(r'Counts')
    #plt.ylim(-0.001, 0.002)
    plt.show()

#-----------------------------------------------------------------------------------------------------------

def plot_continuum_norm(l, s, l_points, s_points, s_norm, poly, l_ca):
    # Initial parameters:
    pref = np.median(poly(l))
    # Plot:
    plt.figure()
    plt.plot(l, s/pref, '-', color='darkgrey', linewidth=0.5)
    for i in range(len(l_ca)): plt.axvline(x=l_ca[i], color='g', linestyle='-.')
    plt.plot(l, poly(l)/pref, 'r--')
    plt.plot(l_points, s_points/pref, 'ro')
    plt.plot(l, s_norm-1, 'k-', linewidth=0.5)
    plt.axhline(0, color='r', linestyle=':')
    # Settings:
    plt.xlabel(r'$\lambda$ (Å)')
    plt.ylabel(r'Norm. Counts')
    plt.show()

def plot_continuum_norm_all(l, s, l_points, s_points, s_norm, poly, l_ca):
    # Initial parameters:
    pref0 = np.median(poly[0](l))
    pref1 = np.median(poly[1](l))
    pref2 = np.median(poly[2](l))
    # Plot:
    fig, ax = plt.subplots()
    plt.plot(l, s/pref1, '-', color='darkgrey', linewidth=0.5, label='Spectrum before')
    #-------
    plt.plot(l, poly[1](l)/pref1, 'r--', label='Continuum peak')
    plt.plot(l, poly[0](l)/pref1, 'r-.', label='Continuum point', alpha=0.28)
    plt.plot(l, poly[2](l)/pref1, 'r:',  label='Continuum mean',  alpha=0.40)
    plt.plot(l_points[0], s_points[0]/pref1, 'ro', ms=4, alpha=0.3)
    plt.plot(l_points[1], s_points[1]/pref1, 'ro', ms=4)
    plt.plot(l_points[2], s_points[2]/pref1, 'ro', ms=4, alpha=0.3)
    #-------
    plt.plot(l, s_norm[1]-1, 'k-', linewidth=0.5, label='Spectrum after')
    plt.axhline(0,         color='g', linestyle='--', label='Continuum')
    plt.axvline(x=l_ca[0], color='g', linestyle=':',  label='Ca lines')
    plt.axvline(x=l_ca[1], color='g', linestyle=':')
    # Legend:
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[2], handles[3], handles[0], handles[4], handles[5], handles[6]]
    labels  = [ labels[1],  labels[2],  labels[3],  labels[0],  labels[4],  labels[5],  labels[6]]
    ax.legend(handles, labels, loc='best', ncol=1, bbox_to_anchor=(1, 0.8))
    # Settings:
    plt.xlabel(r'$\lambda$ (Å)')
    plt.ylabel(r'Norm. Counts')
    plt.ylim(-1, 1.4)
    plt.show()

#-----------------------------------------------------------------------------------------------------------

def plot_sindex_scatter(l, s_dif, s_std0, s_std, bands):
    # Plot:
    plt.figure()
    plt.plot(l, s_dif*100,  '-', color='lightgrey', label=r'scatter($i$)')
    plt.plot(l, s_std0*100, 'k-', linewidth=1.0, label=r'$\sigma_i$')
    plt.plot(l, s_std*100,  'r-', linewidth=1.2, label=r'$\mu_i(\sigma_i)$')
    plt.axhline(0, linestyle=':', color='k')
    plt.axvline(bands[0], linestyle=':', color='b')
    plt.axvline(bands[1], linestyle=':', color='g')
    plt.axvline(bands[2], linestyle=':', color='g')
    plt.axvline(bands[3], linestyle=':', color='r')
    # Settings:
    plt.xlabel(r'$\lambda$ (Å)')
    plt.ylabel(r'Uncertainty (\%)')
    plt.legend(loc='best', ncol=1)
    plt.xlim(min(l), max(l))
    plt.ylim(-10, 50)
    plt.show()

def plot_sindex_bands(l, s, s_tri_K, s_tri_H, K2_indices, H2_indices, K2_fluxes, H2_fluxes, \
                      l_k1_inter, l_k2_inter, l_h1_inter, l_h2_inter, \
                      s_k1_inter, s_k2_inter, s_h1_inter, s_h2_inter, \
                      Kp_wave, Hp_wave, Kp_fluxes, Hp_fluxes, Km_indices, Hm_indices,\
                      K, H, K1_indices, H1_indices):
    s_scale = 1.2
    y = 11
    grid = plt.GridSpec(y, 4, wspace=0.0, hspace=0.0)
    #---------------------
    ax0 = plt.subplot(grid[:int(y/2),:3])
    # K zoom subplot:
    ax0.plot(l, s, 'k-', lw=0.5, label='Spectrum')
    ax0.axvline(l[K1_indices[0]],  c='g', ls='--', label='Bandpass 1.09 Å')
    ax0.axvline(l[K1_indices[-1]], c='g', ls='--')
    ax0.plot(l[K2_indices], s_tri_K,   'g:', lw=1.3, label='Bandpass triangle')
    ax0.plot(l[K2_indices], K2_fluxes, c='deeppink', ls='-', lw=1.2, label='Triangle grid')
    ax0.plot(Kp_wave,       Kp_fluxes, c='b', ls='-', lw=1.2, label='Polygon grid')
    #ax0.plot(l[Km_indices], s[Km_indices], c='deeppink', lw=1.2, label='Mean grid')
    ax0.plot(l_k1_inter, s_k1_inter, 'ro', alpha=0.3, label='Intersections')
    ax0.plot(l_k2_inter, s_k2_inter, 'ro', alpha=0.3)
    # H zoom Subplots:
    ax1 = plt.subplot(grid[int(y/2)+1:,:3])
    ax1.axvline(l[H1_indices[0]],  c='g', ls='--')
    ax1.axvline(l[H1_indices[-1]], c='g', ls='--')
    ax1.plot(l, s, 'k-', lw=0.5)
    ax1.plot(l[H2_indices], s_tri_H,   'g:', lw=1.3)
    ax1.plot(l[H2_indices], H2_fluxes, c='deeppink', ls='-', lw=1.2)
    ax1.plot(Hp_wave,       Hp_fluxes, c='b', ls='-', lw=1.2)
    #ax1.plot(l[Hm_indices], s[Hm_indices], c='deeppink', lw=1.2)
    ax1.plot(l_h1_inter, s_h1_inter, 'ro', alpha=0.3)
    ax1.plot(l_h2_inter, s_h2_inter, 'ro', alpha=0.3)
    # Settings:
    ax0.set_ylim([0, 0.2])
    ax0.set_xlim([K-1.3, K+1.3]) 
    ax1.set_ylim([0, 0.2])
    ax1.set_xlim([H-1.3, H+1.3])
    #ax0.set_xticklabels([])
    ax0.annotate('K bandpass', (K-0.2, 0.02))
    ax1.annotate('H bandpass', (H-0.2, 0.02))
    ax0.set_ylabel('Norma. Counts')
    ax1.set_ylabel('Norma. Counts')
    ax1.set_xlabel(r'$\lambda$ (Å)')
    ax0.legend(loc='upper left', fontsize=12, ncol=1, bbox_to_anchor=(1.0, 0.5))
    plt.show()
    
def plot_sindex_fluxes(l, s, band_indices, fluxes, X):
    V, K, H, R = X[0], X[1], X[2], X[3]
    # Plot:
    fig, ax = plt.subplots()
    s_scale = 1.2
    #------
    ax.fill_between(l[band_indices[0]], fluxes[0], color='b', alpha=0.3)
    ax.fill_between(l[band_indices[1]], fluxes[1], color='r', alpha=0.3)
    ax.fill_between(fluxes[8][0], fluxes[8][1], color='g', alpha=0.5)
    ax.fill_between(fluxes[9][0], fluxes[9][1], color='g', alpha=0.5)
    #------
    ax.plot([l[band_indices[6][ 0]], l[band_indices[8]]], [0, 1], c='g', ls=':')
    ax.plot([l[band_indices[6][-1]], l[band_indices[8]]], [0, 1], c='g', ls=':')
    ax.plot([l[band_indices[7][ 0]], l[band_indices[9]]], [0, 1], c='g', ls=':')
    ax.plot([l[band_indices[7][-1]], l[band_indices[9]]], [0, 1], c='g', ls=':')
    ax.axvline(V, ymax=1/s_scale, c='b', ls=':')
    ax.axvline(R, ymax=1/s_scale, c='r', ls=':')
    #------
    ax.annotate(r'$V$', (V-0.5, 1.07))
    ax.annotate(r'$K$', (K-0.5, 1.07))
    ax.annotate(r'$H$', (H-0.5, 1.07))
    ax.annotate(r'$R$', (R-0.5, 1.07))
    #------
    ax.plot(l, s, 'k-', label='Spectrum', lw=0.3)
    # Settings:
    fig.subplots_adjust(wspace=0, hspace=0.0)
    ax.set_xlabel(r'$\lambda$ (Å)')
    ax.set_ylabel( 'Norm. Counts')
    ax.set_ylim([0, s_scale])
    plt.show()

#########################################################################################################
#                                     PLOTS FOR GENERAL UTILITIES                                       #
#########################################################################################################    

def plot_sky_background(image, disp, yfit_below, yfit_above, yfit_order, l_sky):
    #Plot spectrum with inter-order fits:
    plt.figure()
    FITS(image.T, 'linear')
    plt.plot(disp, yfit_below, 'r-', linewidth='1.5', label='Polynomial fit')
    plt.plot(disp, yfit_above, 'g-', linewidth='1.5', label='Polynomial fit')
    plt.plot(disp, yfit_order.round(), 'm-', linewidth='1.5', label='Polynomial fit')
    plt.xlabel('Dispersion (pixels)'); plt.ylabel('Cross Dispersion (pixels)')
    plt.show()
    # Plot final sky background as function of wavelength:
    plt.figure()
    plt.plot(l_sky, 'k-')
    plt.xlabel('Dispersion (pixels)'); plt.ylabel('Median Counts')
    plt.show()

def plot_locate(t, dif0, dif1, cut_off, n, above, below):
    cut_up = np.ones(len(t))*cut_off
    cut_dw = -cut_up
    plt.figure()
    plt.plot(t, dif0, '-', color='lightgrey', linewidth=1.3, label='Outliers')
    plt.plot(t, dif1, 'k-', label='Corrected', linewidth=0.8)
    plt.plot(t, cut_up, 'r--')
    plt.plot(t, cut_dw, 'r--', label='Cutoff')
    #plt.plot(above, below, 'b+')
    # Print used n:
    xpos = t[0]+(t[-1]-t[0])*0.1
    ypos = max(dif0)-(max(dif0)-min(dif0))*0.1
    plt.text(xpos, ypos, '$n={}$'.format(n), fontsize=15)
    plt.legend(loc='best')
    plt.show()

#########################################################################################################
#                                           PLOTS FOR TEST                                              #
#########################################################################################################    

def plot_blue_moves(x, y, time):
    """ This function makes a plot of the spectral displacement/scatter over time. """
    plt.figure() 
    sc = plt.scatter(x, y, c=time, s=40, linewidth=0.5, edgecolor='k', cmap='rainbow')
    plt.colorbar(sc)
    plt.xlabel('Dispersion (pixel)')
    plt.ylabel('Cross Dispersion (pixel)')
    plt.show()

def plot_rv_stability(time, y, sigma):
    """ This function makes a plot of the spectral displacement/scatter over time. """
    # Find parameters:
    import scipy.constants
    import datetime
    #import pandas as pd
    #--------------
    t = [julian.from_jd(i, fmt='jd') for i in time]       
    lx, sx = 0.052783326947974274 * y, 0.052783326947974274 * sigma
    rv = lx/4000*scipy.constants.c*1e-3
    sigma = sx/4000*scipy.constants.c*1e-3
    # Plot:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(t, y,  'wo')
    ax2.errorbar(t, rv, sigma, None, 'o', ms=3, lw=0.5, c='grey', mec='mediumvioletred', mfc='mediumvioletred')
    #ax2.plot(t, rv, 'o', ms=3.0, c='mediumvioletred')
    #ax2.plot(t, rv, '-', lw=0.3, c='mediumvioletred')
    # Settings:
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dispersion (pixel)')
    ax2.set_ylabel(r'RV drift (km s$^-1$)')
    plt.grid(color='lightgrey')
    #plt.xticks(rotation=-45)
    plt.show()

    
def plot_match_coordinates(array1, array2, indices1, indices2):
    # indices1: array1 coordinates within threshold.
    # indices2: array2 coordinates matching array1 (hence final list)
    # Plotting a histogram showing the best threshold:
    #pt.HIST(dis, 2000, 'Bins', '$\Delta N$', [0.0, 0.0025], [0, 1e3])
    # Plot coordinates and match:
    #plt.figure()
    plt.scatter(array1[:,1], array1[:,0], marker='o', facecolors='none', edgecolors='darkgrey')
    plt.scatter(array2[:,1], array2[:,0], marker='o', facecolors='none', edgecolors='b')
    plt.scatter(array2[:,1][indices2], array2[:,0][indices2], s=1e2, marker='+', edgecolors='r')
    plt.title('{} lines in common out of ({}, {}) available'.format(len(indices2), \
                                                                    len(array1), len(array2)))
    plt.xlabel(r'$x$ (pixel)'); plt.ylabel(r'$y$ (pixel)')
    plt.show()
    
#########################################################################################################
#                                           PRINT TO BASH                                               #
#########################################################################################################

def loading(i, i_max):
    """ This function print the loading status of. """
    sys.stdout.write(u"\u001b[1000D")
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.write(str(i + 1) + "Loding... %")
    sys.stdout.flush()
    
def compilation(i, i_max, text):
    """ This function print out a compilation time menu bar in the terminal. """ 
    percent = (i + 1) / (i_max * 1.0) * 100
    # We here divide by 2 as the length of the bar is only 50 characters:
    bar = "[" + "-" * int(percent/2) + '>' + " " *(50-int(percent/2)) + "] {}% {}".format(int(percent), text)
    sys.stdout.write(u"\u001b[1000D" +  bar)
    sys.stdout.flush()


# hdul = fits.open('file')
# data = hdul[0].data
# hdr = hdul[0].header
# hdr['targname'] = 'NGC121-a'
# hdr[27] = 99
# hdr.set('observer', 'Edwin Hubble')
# fits.writeto('{}TA_{}.fits'.format(self.path, self.date), TA, TA_hdu, overwrite=True)


#box = plt.gca()
#box.add_patch(Rectangle((V-VR/2, axes[0]), VR, axes_ylen, color='w', linestyle=':'))
#box.add_patch(Rectangle((H-HK/2, axes[0]), HK, axes_ylen, color='w', linestyle=':'))
    

########################################

        # # Find the mean, unitless difference, and the std:
        # s_mea  = self.convolve(s[1], 'mean', 3)
        # s_dif  = s[1]/s_mea - 1

        # # First make sure to remove obvious outliers:
        # s_std0 = self.convolve(s_dif, 'std', 100)

        # # Iterate to smoothen out sharp peaks:
        # s_wei = self.convolve(s[1],   'mean', 200)
        # s_std = self.convolve(s_std0, 'mean', 100) * s_wei/np.max(s_wei)
