import numpy as np

from . import songtools


def make_masterbias(biaslist):
    """Combine a set of BIAS frames into the combined BIAS frame.

    For now we use FIEStools method (called median filtering)
    """
    # TODO check if the Bias level varies between images (maybe there is a ramp)??...
    # TODO What about checking for structure??
    # TODO check header info for variations in f.x. ccd temp??
    # TODO implement some kind of robust mean?
    # TODO check flux level / handle scaling

    # Determine the number of bias frames to combine and check if files exist
    nframes = songtools.check_frame_list(biaslist, 'biaslist')

    # TODO should we move datacube creation out of this function?
    # Create a 3D dataset of the bias frames
    datacube = songtools.createdatacube(biaslist, extension=0)  # FIXME extensions????
    if datacube is None:
        raise Exception('Could not process list of BIAS frames - Stopped')

    nsetone = (nframes / 2)  # Important to divide integer by integer
    nsettwo = nframes - nsetone  # so we get (floored-)integer results

    firstmean = np.sum(datacube[0:nsetone, :, :], axis=0, dtype='float64') / float(nsetone)
    secondmean = np.sum(datacube[nsetone:, :, :], axis=0, dtype='float64') / float(nsettwo)

    # Determine the standard deviation (unfiltered RON) of the difference between the frames
    # print('Calculating unfiltered standard deviation of BIAS frames')
    unfilteredRON = np.std(firstmean - secondmean, dtype='float64') * np.sqrt((nsetone * nsettwo) / float(nframes))
    # print('Unfiltered standard deviation is {0} ADU'.format(unfilteredRON))

    del (firstmean, secondmean)

    # Determine the mean bias value
    # print('Calculating unfiltered mean bias level')
    biaslevel = np.mean(datacube, dtype='float64')
    # print('Unfiltered mean bias level is {0} ADU'.format(biaslevel))

    # Set a highest and lowest value for plotting bias histogram
    blow = int(biaslevel - 5.0 * unfilteredRON)
    bhigh = int(biaslevel + 5.0 * unfilteredRON)
    nbins = bhigh - blow

    # In principle, one should do pixel rejection on a pixel by pixel basis,
    # and not use a global average. Still, because one can expect the BIAS
    # to be reasonably flat, one can use the global average, and only reject
    # a handful of pixels too many.
    # If there would be a global structure (for example, a ramp) in the image,
    # one cannot use the global average, but should use 2-d (pixel-by-pixel)
    # averages.
    # biaslevel_mean = np.mean(datacube, dtype='float64', axis = 0 )
    # biaslevel_median = np.median(datacube, axis = 0) # FIXME maybe use bottleneck module instead?
    # Outliers = np.where((biaslevel_mean - biaslevel_median) / unfilteredRON > 2. ) # FIXME when is the difference significant?

    # Set highest and lowest acceptable values for bias pixels
    rlow = biaslevel - 5.0 * unfilteredRON
    rhigh = biaslevel + 5.0 * unfilteredRON

    # bias_list = []
    # bias_sigma_list = []

    # print('Replacing pixel values above and below 5 times standard deviation with the unfiltered mean bias level')
    # Loop to conserve memory (the following requires a lot of memory)
    for i in range(nframes):
        datacube[i, :, :] = np.select([datacube[i, :, :] < rlow, datacube[i, :, :] <= rhigh,
                                       datacube[i, :, :] > rhigh], [biaslevel, datacube[i, :, :], biaslevel])

        # bias_list = np.append(bias_list, np.mean(datacube[i, :, :]))
        # bias_sigma_list = np.append(bias_sigma_list, np.std(datacube[i, :, :]))

    # Check if the Bias level varies between images
    # TODO implement testing this...

    # Determine 'Read-out noise image'
    RON = np.std(datacube, axis=0)
    # Determine Read-out noise level and RMS
    RON_level = np.mean(RON)
    RON_sigma = np.std(RON)

    # print('Measured Read-out noise level is {0:.3f}+/-{1:.3f} ADU'.format(RON_level, RON_sigma))

    # Create an output frame copying a bias frame
    # messageLog.put('Creating combined BIAS frame')

    # outdata = pyfits.open(biaslist[0])  # FIXME handling of header info???
    # dummy = outdata[0].data  # Touch fits data to update header info - FIXME This is a lazy approach! should we do someting else???
    # del (dummy)
    #
    # # Fill the output frame with data and calculate average image
    # outdata[0].data = np.mean(datacube, axis=0, dtype='float64')
    #
    # bias_level = np.mean(outdata[0].data)
    # bias_sigma = np.std(outdata[0].data)

    return np.mean(datacube, axis=0, dtype='float64')


def make_masterflat(flatlist):
    """Combine a set of FLAT frames into the combined FLAT frame.

    The currently implemented method is averaging of the images.
    """
    # TODO implement some kind of robust mean?
    # TODO check flux level / handle scaling

    # Determine the number of bias frames to combine and check if files exist
    nframes = songtools.check_frame_list(flatlist, 'flatlist')

    # Create an output frame copying a flat frame
    # outdata = pyfits.open(flatlist[0], mode='readonly')  # FIXME handling of header info???
    # dummy = outdata[0].data # Touch fits data to update header info - FIXME This is a lazy approach! should we do someting else???
    # del(dummy)

    #  Create a 3D dataset of the flat frames # FIXME handle orientation ??
    datacube = songtools.createdatacube(flatlist, extension=0) # FIXME handle extension
    #
    if datacube is None:
        raise Exception('Could not process list of FLAT frames - Stopped')

    #
    # # FIXME check if flux level change? maybe also check if there is structure? (changes)?
    # # for image in datacube:
    # #     avg = np.mean(image)
    # #     overscan, overscan_std = taskutils.measureoverscan(image, robust=True)
    #
    avg_flat = np.mean(datacube, axis=0, dtype='float64')  # FIXME maybe use robust mean??

    return avg_flat


def make_normflat():
    pass


def make_wavesol():
    pass




