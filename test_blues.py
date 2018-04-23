#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------
TEST FOR 'BLUES' PACKAGES
-------------------------
"""

# Add directories:
import sys
sys.path.append('../blues/')


# Packages:
from click.testing import CliRunner
import blues
import extraction as extr
import cli

###########################################################################################################
#                                            TEST FUNCTIONS                                               #
###########################################################################################################

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'blues.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def system_test():
    import glob
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    import numpy as np

    # Load data:
    path = '/home/nicholas/Data/SONG/blues/testflat1/'
    #path = '/home/jensjh/uni/SONG/Blue_spectrograph/'
    files = np.sort(glob.glob('{}{}*'.format(path, 'bs1')))
    data  = np.median(np.array([pyfits.getdata(str(fil)) for fil in files]), axis=0)
    data = data[:600, :].T

    # Trace orders inside image:
    tr = extr.trace(data, 20, 50, 40, num_orders=4, num_peaks=10)
    inter_mask = extr.make_inter_order_mask(data, 10, tr)

    plt.figure()
    plt.imshow(inter_mask, alpha=0.5)
    plt.imshow(data, vmin=300, vmax=2000, alpha=0.5)

    scatter = extr.remove_background(data, inter_mask)
    plt.figure()
    plt.imshow(scatter)

    dispr = np.arange(data.shape[0])
    poss, order = extr.cut_out_order(np.polyval(tr['order_0'], dispr), data - scatter, 31)

    plt.figure()
    plt.imshow(order, vmax=100)

    norm_order, spec = extr.simple_sum_order(order)

    plt.figure()
    plt.plot(spec)

    kernel = extr.get_kernel_window_size(order.shape[0], 6)
    ron = 10.
    spec_opt, ssp, xopi, xops, normord, mask = extr.optext_full_order(order, poss, ron, kernel)

    plt.figure()
    plt.plot(spec_opt)
    plt.plot(ssp)


    plt.figure()
    plt.plot(poss.T, norm_order.T, 'k.')
    plt.plot(poss.T, xopi.T, 'r.')

    plt.show()
