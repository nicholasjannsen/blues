#-*- coding: utf-8 -*-

"""
TEST FOR 'BLUES' PACKAGES:
--------------------------
"""

# Add directories:
import sys

# Packages:
import glob
import numpy as np
from click.testing import CliRunner

# Modules
#import blues
import extraction as extr
import cli

# Plots:
import matplotlib.pyplot as plt
from astropy.io import fits

###########################################################################################################
#                                            TEST FUNCTIONS                                               #
###########################################################################################################

def test(name):

     if name=='focus':
          # Function
          from FocusingOnTheBlues import FocusingOnTheBlues
          path = '/home/nicholas/Data/SONG/blues/testfocus/'
          XX = FocusingOnTheBlues(path, 'bs1')
          a = XX.focus_using_flats()

     
     if name=='response':

          """Sample pytest fixture.
          See more at: http://doc.pytest.org/en/latest/fixture.html
          """
          # import requests
          # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


     if name=='test_content':
          #test_content(response):
          """Sample pytest test function with the pytest fixture as an argument."""
          # from bs4 import BeautifulSoup
          # assert 'GitHub' in BeautifulSoup(response.content).title.string


     if name=='test_command_line_interface':
          """Test the CLI."""
          runner = CliRunner()
          result = runner.invoke(cli.main)
          assert result.exit_code == 0
          assert 'blues.cli.main' in result.output
          help_result = runner.invoke(cli.main, ['--help'])
          assert help_result.exit_code == 0
          assert '--help  Show this message and exit.' in help_result.output


     if name=='system_test':

          #-----------
          # LOAD DATA:
          #-----------
          
          path0 = '/home/nicholas/Data/SONG/blues/testflat/'
          path1 = '/home/nicholas/Data/SONG/blues/night/'
          
          ### Make master file:
          # files = np.sort(glob.glob('{}{}*'.format(path0, 'bs1')))
          # data  = np.median(np.array([fits.getdata(str(fil)) for fil in files]), axis=0)
          # fits.writeto('{}masterflat.fits'.format(path0), data)
          # sys.exit()

          ### Masterfile:
          data  = fits.getdata('{}masterflat.fits'.format(path0))
          data1 = fits.getdata('{}bs1_2017-10-03T21-17-29.fits'.format(path1))

          # Data clipping:
          data  =  data[100:500, :].T
          data1 = data1[100:500, :].T

          #plt.imshow(data1, vmin=200, vmax=1000)
          #plt.show()
          
          #---------------------------
          # TRACE ORDERS INSIDE IMAGE:
          #---------------------------

          tr = extr.trace(data, 10, 10, 40, 0, 5, 5, 10, plot=0)

          #------------------
          # INTER-ORDER MASK:
          #------------------

          inter_mask = extr.make_inter_order_mask(data, 10, tr, 0, 0, plot=1)
          
          #-----------------------------------
          # REMOVE BACKGROUND/SCATTERED LIGHT:
          #-----------------------------------

          scatter = extr.remove_background(data, inter_mask, 5, 5, 1000, 5, plot=1)
          
          #---------------
          # CUT OUT ORDER:
          #---------------

          dispr = np.arange(data.shape[0])
          poss, order = extr.cut_out_order(np.polyval(tr['order_2'], dispr), data - scatter, 31, plot=0)
          norm_order, blaze = extr.simple_sum_order(order)
          
          ############### Star

          scatter1 = extr.remove_background(data1, inter_mask, plot=0)
          poss, order = extr.cut_out_order(np.polyval(tr['order_2'], dispr), data1, 31, plot=0)

          plt.figure()
          plt.plot(order.sum(axis=1)/blaze)
          plt.show()

          
     # if name=='system_test_class':
     #      path = '/home/nicholas/Data/SONG/blues/testflat/'
     #      data  = fits.getdata('{}masterflat.fits'.format(path))
     #      from extraction_class import extraction
     #      ext = extraction(data, )

     #      #---------------------------
     #      # TRACE ORDERS INSIDE IMAGE:
     #      #---------------------------
          
     #      tr = extr.trace(data, 20, 50, 40, 0, 5, 4, 10, 0)

     #      #-----------------------
     #      # MAKE INTER-ORDER MASK:
     #      #-----------------------
          
     #      inter_mask = extr.make_inter_order_mask(data, 10, tr, 1, 2, 0)

     #      #-----------------------------------
     #      # REMOVE BACKGROUND/SCATTERED LIGHT:
     #      #-----------------------------------
          
     #      scatter = extr.remove_background(data, inter_mask, plot=0)
          
     #      #---------------
     #      # CUT OUT ORDER:
     #      #---------------

     #      dispr = np.arange(data.shape[0])
     #      poss, order = extr.cut_out_order(np.polyval(tr['order_0'], dispr), data - scatter, 31)
          
     #      plt.figure()
     #      plt.imshow(order, vmax=100)
     #      plt.show()
          
     #      #sys.exit()
          
     #      norm_order, spec = extr.simple_sum_order(order)
          
     #      plt.figure()
     #      plt.plot(spec)
          
     #      kernel = extr.get_kernel_window_size(order.shape[0], 6)
     #      ron = 10.
     #      spec_opt, ssp, xopi, xops, normord, mask = extr.optext_full_order(order, poss, ron, kernel)
          
     #      plt.figure()
     #      plt.plot(spec_opt)
     #      plt.plot(ssp)
          
          
     #      plt.figure()
     #      plt.plot(poss.T, norm_order.T, 'k.')
     #      plt.plot(poss.T, xopi.T, 'r.')
          
     #      plt.show()
          
          
if __name__ == '__main__': #---------------------------- Main function --------------------------#
     # FUNCTION CALL:
     test(name='system_test')    
