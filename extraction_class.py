# -*- coding: utf-8 -*

import numpy as np
import sys
import matplotlib.pyplot as plt
import bottleneck as bn
from skimage import feature as skfeature

###############################################################################################################
#                                               DEFINE CLASS                                                  #
###############################################################################################################

class extraction(object):
    # INITILIZE THE CLASSE: 
    def __init__(self, data, plot):

        #-------------------------------
        # DEFINE GLOBAL VARIABLES (DGV):
        #-------------------------------

        # Customized information:
        self.data = data   # Name of images
        self.plot = plot       # Plot if 1
                
        # Image dimensions information:
        self.h = 2200       # Heigth of image
        self.w = 2750       # Dispersion pixel length (direction along the orders) 

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
    #                                              FIND ORDERS                                                #
    ###########################################################################################################

    def trace(smooth_win, exclude_border, min_distance, threshold_abs=0, trace_tol=5, \
              num_orders=None, num_peaks=None, plot=None):
        """
        This function find the orders in an eshel spectrum by tracing the maximum. . 
        ----------------------------
                    INPUT          :
        ----------------------------
        image_data     (2d array)  : A single image. 
        smooth_win     (int, float): Smooth value to enhance orders.
        exclude_border (int, float): Border edges that should be exluded.
        min_distance   (int, float): 
        threshold_abs  (int, float):
        num_orders     (int, float):
        num_peaks      (int, float):
        trace_tol      (int, float):
        ----------------------------
                   OUTPUT          :
        ----------------------------
        order_traces  ()
        """

        # Image constants:
        center_disp = int(self.data.shape[0] / 2)               # Center position of dispersion axis
        reference_interval = [center_disp - 5, center_disp + 6]  # Central position interval 

        # FUNCTION CALL: 
        reference_order_center_pos = find_reference_order_centers(image_data, reference_interval, smooth_win, \
                                                                  exclude_border, min_distance, \
                                                                  threshold_abs=threshold_abs, \
                                                                  num_peaks=num_orders)

        # FUNCTION CALL:
        ridge_pos_cross, ridge_pos_disp = find_order_ridges(image_data, smooth_win, exclude_border, min_distance,\
                                                            threshold_abs=threshold_abs, num_peaks=num_peaks)

        # FUNCTION CALL:
        order_traces = {}
        for i, order_pos in enumerate(np.sort(reference_order_center_pos)[::-1]):

            order_trace_cross, order_trace_disp = trace_order(center_disp, order_pos[0], ridge_pos_disp, \
                                                              ridge_pos_cross, min_distance, \
                                                              disp_gap_tolerance=trace_tol)
            #plt.plot(order_trace_cross, order_trace_disp)
            poly_coefs = np.polyfit(order_trace_disp, order_trace_cross, 5)
            order_traces['order_{}'.format(i)] = poly_coefs

        # Plot if you like:
        if plot==1:
            plt.plot(ridge_pos_cross, ridge_pos_disp, 'b.')
            plt.plot(reference_order_center_pos, reference_order_center_pos*0 + center_disp, 'rs')
            plt.show()

        return order_traces




    def find_reference_order_centers(image_data, reference_interval, smooth_win, exclude_border, min_distance, \
                                     threshold_abs=0, num_peaks=None):
        """
        This function finds the center position of an order used as reference.  
        """

        # Collapse in disp direction to reduce cosmic ray contamination:
        # (FIXME done to make this robust against cosmics - maybe it is not needed)
        center_rows_median = np.median(image_data[reference_interval[0]:reference_interval[1], :], axis=0)

        # Smooth cross_dispersion direction to prepare for the peak-detection algorithm:
        center_row_median_convolved = bn.move_sum(center_rows_median.astype(np.float), smooth_win, min_count=1)

        # Find orders using a peak detection function from scikit-image:
        order_centres = skfeature.peak_local_max(center_row_median_convolved, exclude_border=exclude_border,\
                                                 min_distance=min_distance, threshold_rel=0,\
                                                 threshold_abs=threshold_abs, num_peaks=num_peaks)
        return order_centres - int(smooth_win/2)




    def find_order_ridges(image_data, smooth_win, exclude_border, min_distance, threshold_abs=0, num_peaks=None):
        """
        This function find ridges.  
        """

        ridge_indices_cross = []
        ridge_indices_disp = []

        for i, crossorder in enumerate(image_data):

            # TODO should smoothing be handled separately?
            top_hat_conv = bn.move_sum(crossorder.astype(np.float), smooth_win, min_count=1)  
            peaks = skfeature.peak_local_max(top_hat_conv, exclude_border=exclude_border,\
                                             min_distance=min_distance, threshold_rel=0, \
                                             threshold_abs=threshold_abs, indices=True, num_peaks=num_peaks)
            peaks -= int(smooth_win/2)
            ridge_indices_disp = np.append(ridge_indices_disp, np.ones(peaks.shape[0]) * i)
            ridge_indices_cross = np.append(ridge_indices_cross, peaks)

        return ridge_indices_cross, ridge_indices_disp




    def trace_order(x_reference_index, y_reference, all_orders_x, all_orders_y, order_width, disp_gap_tolerance=5):
        """
        This function traces all spectroscopic orders inside the data image
        INPUT:
        - x_refe
        """
        x = np.unique(all_orders_x)
        cross_disp = []
        disp = []
        cross_disp_gap_tolerance = int(order_width/2.)

        y_last = y_reference # If center_row is not an integer this will fail!
        x_last = x[x_reference_index]
        for xi in x[x_reference_index:]:
            index_xi = all_orders_x == xi
            orders_y = all_orders_y[index_xi]

            min_dist_index = np.argmin(np.abs(orders_y - y_last))
            new_y_pos = orders_y[min_dist_index]

            if (np.abs(new_y_pos - y_last) < cross_disp_gap_tolerance) & \
               (np.abs(xi - x_last) < disp_gap_tolerance):
                cross_disp.append(new_y_pos)
                y_last = cross_disp[-1]
                disp.append(xi)
                x_last = disp[-1]

        y_last = y_reference # If center_row is not an interger this will fail!
        x_last = x[x_reference_index]
        for xi in x[x_reference_index-1::-1]:
            index_xi = all_orders_x == xi
            orders_y = all_orders_y[index_xi]

            min_dist_index = np.argmin(np.abs(orders_y - y_last))
            new_y_pos = orders_y[min_dist_index]

            if (np.abs(new_y_pos - y_last) < cross_disp_gap_tolerance) & \
               (np.abs(xi - x_last) < disp_gap_tolerance):
                cross_disp.append(new_y_pos)
                y_last = cross_disp[-1]
                disp.append(xi)
                x_last = disp[-1]

        index = np.argsort(disp)
        return np.array(cross_disp)[index], np.array(disp)[index]


    ###########################################################################################################
    #                                           INTER-ORDER MASK                                              #
    ###########################################################################################################

    def make_inter_order_mask(image_data, order_width, order_traces, low_nudge=1., high_nudge=2., plot=0):
        """
        This function find the orders in an eshel spectrum by tracing the maximum. . 
        ----------------------------
                    INPUT          :
        ----------------------------
        image_data     (2d array)  : A single image. 
        smooth_win     (int, float): Smooth value to enhance orders.
        exclude_border (int, float): Border edges that should be exluded.
        min_distance   (int, float): 
        threshold_abs  (int, float):
        num_orders     (int, float):
        num_peaks      (int, float):
        trace_tol      (int, float):
        ----------------------------
                   OUTPUT          :
        ----------------------------
        order_traces  ()
        """

        inter_order_mask = image_data * 0 + 1   # Initial image mask of ones 
        disp_length = image_data.shape[0]       # Length of dispersion axis
        disp = np.arange(disp_length)           # Number pixel interval for dispersion direction [0, 2750]
        order_no = sorted(order_traces.keys())  # Orders numbers (string)
        cross_order_center = []                        

        for order in order_no:
            # Get the coefficients from the trace function:
            coefs = order_traces[order]                    
            cross_order_position = np.polyval(coefs, disp) 
            cross_order_center = np.append(cross_order_center, cross_order_position[int(disp_length/2)])
            #plt.plot(cross_order_position); plt.show()
            for disp_i in range(disp_length):
                lower_order_edge = int(np.round(cross_order_position[disp_i] - order_width/2 - low_nudge))
                upper_order_edge = int(np.round(cross_order_position[disp_i] + order_width/2 + high_nudge))
                #print(lower_order_edge)
                inter_order_mask[int(disp_i), lower_order_edge:upper_order_edge] = 0

        inter_order_size = cross_order_center[1:] - cross_order_center[:-1] - order_width - low_nudge - high_nudge

        #-----------------------
        # REMOVE 'GHOST ORDERS':
        #-----------------------

        # Predict inter_order_size:
        xx = np.arange(len(cross_order_center)-1)
        inter_order_size_fit = np.polyfit(xx, inter_order_size, 2)

        size_before = np.polyval(inter_order_size_fit, -1)
        size_after = np.polyval(inter_order_size_fit, len(cross_order_center))

        # Remove 'ghost orders' before first order:
        coefs = order_traces[order_no[0]]
        cross_order_position = np.polyval(coefs, disp)
        for disp_i in range(disp_length):
            lower_inter_order_edge = np.round(cross_order_position[disp_i] - order_width/2 - low_nudge - \
                                              size_before)
            if lower_inter_order_edge < 0: lower_inter_order_edge = 0
            inter_order_mask[disp_i, :lower_inter_order_edge] = 0

        # Remove 'ghost orders' after last order:
        # coefs = order_traces[order_no[-1]]
        # cross_order_position = np.polyval(coefs, disp)
        # for disp_i in range(disp_length):
        #     upper_inter_order_edge = np.round(cross_order_position[disp_i] + order_width/2 + high_nudge + \
        #                                       size_after)
        #     inter_order_mask[disp_i, upper_inter_order_edge:] = 0

        if plot==1:
            plt.figure()
            plt.imshow(inter_order_mask, alpha=1)
            plt.imshow(image_data, vmin=300, vmax=2000, alpha=0.5)
            plt.show()

        return inter_order_mask


    ###########################################################################################################
    #                                         REMOVE SCATTERED LIGHT                                          #
    ###########################################################################################################


    def remove_background(image_data, mask, poly_order_x=4, poly_order_y=4, nsteps=1000, medianfiltersize=5, \
                          orderdef=None, plot=0):
        """
        This function remove the background flux / scattered light. It uses the 'mask' to perform this removal. 
        ----------------------------
                    INPUT          :
        ----------------------------
        image_data       (2d array): A single image 
        mask             (2d array): Background mask with ones and zeros
        poly_order_x   (int, float): Order of polynomy to fits background flux in x 
        poly_order_y   (int, float): Order of polynomy to fits background flux in y
        nsteps         (int, float): Number of steps 
        orderdef       (int, float):
        ----------------------------
                   OUTPUT          :
        ----------------------------
        background_image (2d array):  
        """

        if nsteps <= poly_order_y:
            poly_order_y = int(nsteps / 2)

        (ysize, xsize) = image_data.shape
        background_image = np.zeros((ysize, xsize),  dtype=np.float64)
        xfitarr = np.zeros((nsteps, xsize), dtype=np.float64)

        xx = np.arange(xsize, dtype=np.float64)
        yy = np.arange(ysize, dtype=np.float64)

        ystep = int(ysize / (nsteps - 1))
        yvals = (np.arange(nsteps) * ystep).round()
        ycount = 0

        #--------------------
        # FIT IN Y-DIRECTION:
        #--------------------

        for yind in yvals:

            ymin_ind = np.max([yind-medianfiltersize, 0])
            ymax_ind = np.min([yind+medianfiltersize, ysize-1])

            meanvec = np.average(image_data[ymin_ind:ymax_ind, :], axis=0)

            order_throughs = np.where(mask[yind, :] == 1)[0]

            niter = 0

            # Perform fitting with sigma-clipping
            while 1:

                coefs = np.polyfit(order_throughs, meanvec[order_throughs], poly_order_x)
                xfit = np.polyval(coefs, order_throughs)

                sigma = (meanvec[order_throughs] - xfit) / np.std(meanvec[order_throughs] - xfit)
                rejected = np.extract(sigma > 3, order_throughs)
                order_throughs = np.extract(sigma < 3, order_throughs)

                niter = niter + 1
                if niter == 5 or rejected.size == 0:
                    break

            xfit = np.polyval(coefs, xx)
            xfitarr[ycount, :] = xfit
            ycount = ycount + 1

            # Debugging plot in x-direction (across the orders)
            if 0:
                plt.plot(xx, meanvec, color='green')
                plt.plot(xx, xfit, color='blue')
                plt.plot(order_throughs, meanvec.take(order_throughs))
                plt.show()
            if 0:
                plt.plot(order_throughs, meanvec[order_throughs] - np.polyval(coefs, order_throughs))
                plt.show()

        #--------------------
        # FIT IN X-DIRECTION:
        #--------------------

        for xind in np.arange(xsize):

            # Perform fitting with sigma-clipping
            niter = 0
            goodind = np.arange(nsteps)

            while 1:
                coefs = np.polyfit(yvals.take(goodind), xfitarr[goodind, xind], poly_order_y)
                yfit = np.polyval(coefs, yvals[goodind])

                sigma = (xfitarr[goodind, xind] - yfit) / np.std(xfitarr[goodind, xind] - yfit)
                rejected = np.extract(sigma > 3, goodind)
                goodind = np.extract(sigma < 3, goodind)

                niter = niter + 1
                if niter == 3 or rejected.size == 0 or goodind.size == 0:
                    break

            if goodind.size == 0:
                print("Error: no points left when y-fitting the background")
                coefs=np.polyfit(xfitarr[:, xind])
            background_image[:, xind] = np.polyval(coefs, yy)

            # Debugging plot:
            if 0:
                plt.plot(yy, background_image[:, xind], color='blue')
                plt.plot(yvals, xfitarr[:, xind], color='red')
                plt.plot(yvals[goodind], xfitarr[goodind, xind], color='green')
                plt.show()

        # Plot if you like:
        if plot==1:
              plt.imshow(background_image)
              plt.colorbar()
              plt.show()

        return background_image


    ###########################################################################################################
    #                                         REMOVE SCATTERED LIGHT                                          #
    ###########################################################################################################


    def cut_out_order(order_trace, data, cross_order_width):

        half_width = int(cross_order_width/2.)
        dispersion_length = data.shape[0]

        print(dispersion_length); sys.exit()

        order = np.zeros((dispersion_length, cross_order_width))
        cross_order_positions = np.zeros((dispersion_length, cross_order_width))

        for d in np.arange(dispersion_length):
            position = order_trace[d]
            rounded_position = int(np.round(position))

            cp = data[d, rounded_position - half_width:rounded_position + half_width + 1]
            x = np.arange(-half_width, half_width + 1) - position + rounded_position

            order[d,:] = cp
            cross_order_positions[d, :] = x

            print(d)
            print(order[d,:])
            #sys.exit()

        return cross_order_positions, order


    def cut_out_mask(order_trace, mask, cross_order_width):

        half_width = int(cross_order_width/2.)

        dispersion_length = mask.shape[0]

        new_mask = np.zeros((dispersion_length, cross_order_width))

        for d in np.arange(dispersion_length):
            position = order_trace[d]
            rounded_position = int(np.round(position))

            mask_row = mask[d, rounded_position - half_width:rounded_position + half_width + 1]

            new_mask[d, :] = mask_row

        return new_mask


    def simple_sum_order(order):

        normalized_order = np.zeros(order.shape)
        spectrum = order.sum(axis=1)

        for c in range(order.shape[1]):
            normalized_order[:, c] = order[:, c] / spectrum

        return normalized_order, spectrum


    def decompose_order(order, read_out_noise, xop_weight, spectrum, mask):

        normalized_order = np.zeros(order.shape)

        for d in range(order.shape[0]):

            cp = order[d, :]
            xmask = mask[d, :] * 1.

            weights = xop_weight[d, :]

            variance = spectrum[d] * weights + read_out_noise ** 2.

            spectrum[d] = np.sum(xmask * cp * (weights / variance)) / np.sum(xmask * np.square(weights) / variance)

            normalized_order[d, :] = cp / spectrum[d]

        return normalized_order, spectrum


    def calc_cross_order_profile(order_positions, normalized_order, kernel_window_size, sigma_cut=5):
        # TODO Maybe median or even biweight location could be used instead??
        # TODO Should already calculated mask be used ??
        sorted_ind = np.argsort(np.ravel(order_positions))  # FIXME it might cause problems maybe use resize/reshape instead
        xops = np.ravel(normalized_order)[sorted_ind]
        xops_x = np.ravel(order_positions)[sorted_ind]
        m_arg = bn.move_mean(xops, kernel_window_size)
        m_std = bn.move_std(xops, kernel_window_size)
        bin_center = bn.move_mean(xops_x, kernel_window_size)

        cop_interp = np.interp(xops_x, bin_center[kernel_window_size:], m_arg[kernel_window_size:])
        std_interp = np.interp(xops_x, bin_center[kernel_window_size:], m_std[kernel_window_size:])
        residuals = xops - cop_interp
        sigma = residuals / std_interp

        clip = np.abs(sigma) > sigma_cut

        while clip.any():

            keep = np.abs(sigma) < sigma_cut
            xops = np.extract(keep, xops)
            xops_x = np.extract(keep, xops_x)

            m_arg = bn.move_mean(xops, kernel_window_size)
            m_std = bn.move_std(xops, kernel_window_size)

            bin_center = bn.move_mean(xops_x, kernel_window_size)
            cop_interp = np.interp(xops_x, bin_center[kernel_window_size:], m_arg[kernel_window_size:])
            std_interp = np.interp(xops_x, bin_center[kernel_window_size:], m_std[kernel_window_size:])

            residuals = xops - cop_interp
            sigma = residuals / std_interp
            clip = np.abs(sigma) > sigma_cut

        cop_image = np.interp(order_positions, bin_center[kernel_window_size:], m_arg[kernel_window_size:])
        std_image = np.interp(order_positions, bin_center[kernel_window_size:], m_std[kernel_window_size:])

        cop_image[cop_image < 0.] = 0.  # FIXME include check?

        # TODO maybe optimize???
        cop_area = np.sum(cop_image, axis=1)
        for i, area in enumerate(cop_area):
            cop_image[i, :] /= area

        return cop_image, std_image


    def calc_mask(normalized_order, xop_image, std_image, order, spectrum, read_out_noise, sigma_clip=5):

        sigmas = np.abs(normalized_order - xop_image) / std_image
        mask = np.ones(order.shape)

        for i in range(order.shape[0]):

            sigma_row = sigmas[i, :]

            while np.max(sigma_row * mask[i, :]) > sigma_clip:
                ind = np.argmax(sigma_row * mask[i, :])
                mask[i, ind] = 0

                cp = order[i, :]

                weights = xop_image[i, :]

                variance = spectrum[i] * weights + read_out_noise ** 2.

                spectrum[i] = np.sum(mask[i, :] * cp * (weights / variance)) / np.sum(mask[i, :] * np.square(weights) / variance)

                cross_order_profile = cp / spectrum[i]

                sigma_row = np.abs(cross_order_profile - weights) / std_image[i, :]

        return mask


    def get_kernel_window_size(swarth_length, pixel_over_sampling_factor):
        # kernel_window_size = int(swarth_length * cross_order_width / (cross_order_width * pixel_over_sampling_factor))
        kernel_window_size = int(swarth_length / pixel_over_sampling_factor)
        if np.remainder(kernel_window_size, 2) == 0.:
            kernel_window_size += 1
        return int(kernel_window_size)


    def optext_full_order(order, order_positions, read_out_noise, kernel_window_size, niter=10):

        normalized_order, spectrum = simple_sum_order(order)
        simple_sum_spectrum = spectrum.copy()
        xop_old = np.zeros(order.shape)
        for k in range(niter):
            xop_image, xop_std_image = calc_cross_order_profile(order_positions, normalized_order, kernel_window_size)

            mask = calc_mask(normalized_order, xop_image.copy(), xop_std_image, order, spectrum, read_out_noise)

            normalized_order, spectrum = decompose_order(order, read_out_noise, xop_image, spectrum, mask)
            if np.sum(xop_image - xop_old) < 1.e-8:  # FIXME improve criterion
                break
            xop_old = xop_image.copy()

        return spectrum, simple_sum_spectrum, xop_image, xop_std_image, normalized_order, mask
