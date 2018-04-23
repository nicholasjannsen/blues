# coding=utf-8

import os
import datetime
import numpy as np
import astropy.io.fits as pyfits


############# FITS header related util functions ############


def generate_obslog(directory, startswith=None):
    """Generates an 'obslog' dictionary containing all header info for all fits files in a directory

    The fits header keywords are the keys in the in the dictionary
    and each key item is a list (np.array) with the keyword values for all fits files.

    OBS: The headers in the fits files must have exactly the same keywords as the first listed file.
    Otherwise they are not included in the obslog dictionary.

    Example:
    obslog = generate_obslog('/path/to/directory/')
    obslog.keys() # show header keywords
    obslog['filename'] # print names of all found fits files

    Parameters
    ----------
    :param directory: dictionary containing all header info for fits files in a directory
                    where the fits header keywords are the keys and lists with the keyword values for all fits files
                    are the connected items
    :type directory: string
    :param startswith: A string specifying the prefix, files in the directory should start with
    :type startswith: str

    Returns
    -------
    :return obslog: dictionary containing all header keyword values for fits files in a directory
    :rtype obslog: dict
    """
    # TODO move this function to another module
    # TODO add possibility to use a specified header template

    # List all file in a directory ending with '.fits' and save the absolute path
    extension = 'fits'
    if isinstance(startswith, str):
        fits_list = [os.path.join(directory, f) for f in sorted(os.listdir(directory))
                     if os.path.isfile(os.path.join(directory, f)) & f.startswith(startswith)]
    else:
        fits_list = [os.path.join(directory, f) for f in sorted(os.listdir(directory))
                     if os.path.isfile(os.path.join(directory, f)) & f.endswith(extension)]

    # Use the first frame in the list to generate the dictionary structure
    fits_file = fits_list.pop(0)
    obslog = {'filename': np.array([])}  # Use np.array to be able to filter the lists later
    obslog['filename'] = np.append(obslog['filename'], os.path.abspath(fits_file))
    header = pyfits.getheader(fits_file)
    header_keys = header.keys()
    for key in header_keys:
        obslog[key] = np.array([])
        obslog[key] = np.append(obslog[key], header.cards[key][1])

    # Run through the remaining fits files and append keyword values to the dictionary items
    for fits_file in fits_list:
        header = pyfits.getheader(fits_file)
        if header.keys() != header_keys:
            print('WARNING! Header in fits file {0} does not have the same keywords as the template - SKIPPING FILE'
                  .format(os.path.basename(fits_file)))
            continue
            # raise Exception('Headers in fits files from directory does not have the same keyword content')

        # Append header keyword values to the item list in each key
        obslog['filename'] = np.append(obslog['filename'], os.path.abspath(fits_file))
        for key in header_keys:
            obslog[key] = np.append(obslog[key], header.cards[key][1])

    return obslog


def generate_fits_list(obslog, keywords, keyword_filters):
    """Generates a list of files from the obslog dictionary using a set of keyword+filter pairs

    See the function 'generate_obslog' for an explanaition of the obslog dictionary

    Parameters
    ----------
    :param obslog: dictionary containing all header info for fits files in a directory
                    where the fits header keywords are the keys and lists with the keyword values for all fits files
                    are the connected items
    :type obslog: dict
    :param keywords: boolean specifying choose whether or not to use robust statistics
    :type keywords: boolean

    Returns
    -------
    :return filtered_list: list of files containing one or more specified keyword+filter pairs
    :rtype filtered_list: np.array
    """
    # TODO implementing other logical operators than ==

    filter_string = ''
    for keyword, keyword_filter in zip(keywords, keyword_filters):
        filter_string += keyword + '=' + str(keyword_filter) + ', '
    filter_string = filter_string[:-2]

    # Make boolean array to be able to run over several filter criteria
    filter_list = np.array([True]*np.size(obslog['filename']))

    for keyword, keyword_filter in zip(keywords, keyword_filters):
        # Sanity check
        if not obslog.has_key(keyword):
            raise Exception('The headers do not contain the keyword: "{0}"'.format(keyword))

        # Add filter to the boolian array
        filter_list &= (obslog[keyword] == keyword_filter)

        if not any(filter_list):
            raise Exception('There was no header keyword containing the value: "{0}" of type: {1} '.format(keyword_filter, type(keyword_filter))) # FIXME should an exception be raised?

    filtered_list = obslog['filename'][filter_list]

    return filtered_list


def check_frame_list(framelist, list_name):

    try:
        nframes = len(framelist)
    except TypeError:
        raise Exception("{} does not contain a list".format(list_name))

    for frame in framelist:
        if not os.path.isfile(frame):
            raise Exception("Cannot find the frame {}".format(frame))

    return nframes


def get_date_from_header(header, obs_begin=None):
    """Returns a start of observing night date string in YYYYMMDD format generated from the header['DATE-OBS'] time-string.

    If the header['DATE-OBS'] time-string is earlier than 'obs_begin' the returned date string will be bumped back one day.

    .. Example:
    >hdr = pyfits.getheader('example.fits')
    >hdr['DATE-OBS']
    '2015-03-04T11:39:04.670304'
    >get_date_from_header(hdr, obs_begin=13)
    '20150303'

    Parameters
    ----------
    :param header: Fits file header
    :type header: dict
    :param obs_begin: The hour in utc the (next) observing night begins. Set to 12 if not specified.
    :type obs_begin: int/float

    Returns
    -------
    :return date: Observing night start date in YYYYMMDD format
    :rtype date: string
    """

    if obs_begin is None:
        obs_begin = 12

    try:
        # Assume the DATE-OBS time-string format in YYYY-MM-DDTHH:MM:SS.S
        dt = datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        raise Exception("The header['DATE-OBS'] time-string format is NOT: YYYY-MM-DDTHH:MM:SS.SSSSSS")

    # Check if the hour is above 12
    if dt.hour < obs_begin:
        # if time is after midnight and before 12 noon - Bump the day one back
        dt -= datetime.timedelta(days=1)

    date = dt.strftime("%Y%m%d")  # save datestamp as string with YYYYMMDD format

    return date


def stamp_filename(filename, info):
    """Adds a string to the end of a filename before the extension

    An '_' is prepended to the stamp and if the argument 'info' is a list of strings the strings are joined
    with an '_' is added in between all strings

    Parameters
    ----------
    :param filename: The name of a file possibly with full absolute path
    :type filename: string
    :param info: A (list of) string(s) to the end of the root of a filename
    :type info: string or list

    Returns
    -------
    :return stamped_filename: Filename with info stamped to the end of its root
    :rtype stamped_filename: string
    """
    # TODO move this function to another module
    # make file stamp from list of strings or one string
    # if type(info) is list:
    if isinstance(info, list):
        stamp = '_'+'_'.join(info)
    else:
        stamp = '_'+info

    # split extension from base name and path to file
    base_name, file_extension = os.path.splitext(filename)

    # Add stamp to filename
    stamped_filename = base_name + stamp + file_extension

    return stamped_filename

################################################################################


def get_file_list(directory, prefix='', surfix='.fits'):
    """
    Creates list of files in the specified directory, with the specified extension.

    Parameters
    ----------
    :param directory: directory where to create file list
    :type directory: list
    :param surfix: surfix of the files to be listed
    :type surfix: string
    :param prefix: prefix of the files to be listed
    :type prefix: string


    Returns
    -------
    :return file_names: Broadening function smoothed with Gaussian
    :rtype file_names: list
    """
    file_names = [os.path.join(directory, file_name) for file_name in sorted(os.listdir(directory))
                  if os.path.isfile(os.path.join(directory, file_name)) & file_name.startswith(prefix) & file_name.endswith(surfix)]
    return file_names

################################################################################


def createdatacube(framelist, extension=0):
    """Creates a datacube (3-dimensional array) from a list of images

    Parameters
    ----------
    :param framelist: file list
    :type framelist: list
    :param extension: extension of the files to be listed
    :type extension: string

    Returns
    -------
    :return datacube: datacube (3-dimensional array) containing images
    :rtype datacube: np.ndarray
    """

    length = len(framelist)

    # FIXME handle orientation - if we want to do that move it outside into tasks!
    # flipframe(image, settings['frameorientation'])
    frame = framelist[0]
    # Read a frame
    try:
        data = pyfits.getdata(frame, extension=extension)
    except IOError:
        raise Exception('Cannot open {}'.format(frame))

    # Preallocate array
    datacube = np.empty((length, data.shape[0], data.shape[1]))
    datacube[0, :, :] = data


    # Loop over remaining frames (start from index 1, not 0)
    for i, frame in enumerate(framelist[1:]):

        # Read a frame
        try:
            data = pyfits.getdata(frame, extension=extension)
        except IOError:
            raise Exception('Cannot open {}'.format(frame))

        datacube[i+1, :, :] = data

        if not datacube.shape[1:] == data.shape:
            raise Exception('Data array dim ({0}) in {1} does not match {2} '.format(frame, data.shape, datacube[i+1].shape[1:]))


    del(data)

    # Return the cube
    return datacube



