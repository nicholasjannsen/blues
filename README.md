# A BlueSONG: Tracing magnetic activity in the Ca II lines of solar-type stars using eShel, a commercial spectrograph mounted on the Hertzsprung SONG telescope
This project involves the commercial échelle spectrograph, eShel, mounted on the fully robotic Hertzsprung SONG telescope on Tenerife, to prove the concept of using an affordable spectrograph, to trace stellar magnetic activity in the Ca II lines of solar-type stars. The eShel is a fiber-fed échelle spectrograph with a spectral resolving power of $R \approx 10,000$, and it has proven to be a useful tool for the detection of radial velocities down to \SI{80}{\metre\per\second}, but with the difficulty of receiving a sufficient emission excess in the near-UV, this project is to our knowledge the first of its kind. Since the strongest spectral features observable from ground are the \cahk{} lines, and the fact that long-term observations of stellar chromospheric activity primarily comes from the Mt Wilson 30--year HK-Project, this research likewise use the \cahk{} lines as a proxy of stellar magnetic activity. The project objectives are thus; first to investigate that an off-the-shelf spectrograph like eShel can provide sufficient \cahk{} line emission by observations. Secondly, to calculate the chromospheric activity proxy $S$ and, thirdly, evaluate the success off monitoring activity cycles of stars similar to the Sun using eShel. To meet these requirements, observations obtained with eShel, were supplemented by simultaneous observations made with the FIES spectrograph at the Nordic Optical Telescope, and both dataset was directly compared to the Mt Wilson survey. In a near-future with an increasing number of larger ground and space based telescopes, such as the TMT, ELT, PLATO, JWST, etc., an increase of small and fully robotic telescopes will be needed to complement monitoring, survey, and follow-up observations, of which we begin to see their infancy. Hence, the fact that this master's thesis demonstrates, that the eShel spectrograph can be used to successfully monitor stellar magnetic activity of solar-type stars, confirms that commercial spectrographs may be of significant importance, for future networks of large surveys within the scope of observing and understanding stellar activity in solar-type stars. If proving worthy in terms of efficiency, eShel will provide a possibility and great potential for observing stellar cycles using the Hertzsprung SONG telescope. 

# blues
`Software for` 

# Software description: `blues.py` and `BlueSONG`
This python class is specifically made for the spectroscopic data reduction of the Shelyak eShel spectrograph
which is installed at the Hertzsprung SONG node telescope at Tenrife, Spain. The software is originally built
from structures of the 'SONGWriter' which is SONG's spectroscopic data reduction pipeline, and by others is 
inspired by the data reduction pipeline 'FIESTools' of the FIES spectrograph at the NOT on La Palma.

To explain the software in more detail we here present the results from it. The following code assumes that all data is placed in the same folder, and the science, flat, bias, and dark frames are likewise called so (you can call them whatever you wnat, just remember to change the names of the input files). The file 'test.py' can be used to make an easy test of the software. The following code example illustartes the usage:

```

```

For the photometry software the first 2 entries in 'aperture_photometry' are the stellar coordinates. The next entry is the     "aperture" entry that takes 5 arguments: ['aperture', a, b, q, phi]. Here 'aperture' is either 'ellipse' or 'trace' corresponding to the two apertures. Because we are working with startrails, 'phi' is a tilt angle of the aperture between 0-180 degrees defined by the zero-point of the unit circle (hence counter clockwise from first quadrant). 

Using the ellipse as aperture "a" is the semi-minor axis of the ellipse, "b" is semi-major axis of the ellipse, "q" is the width of the local background flux from the ellipse. 

As mentioned the trace aperture is a mission-specific aperture that uses a circular aperture of radius "a". Given the coordinates of the most left part of the startrail, the COF is found inside this initial circular aperture. Next the circular aperture will be moved one pixel at a time in either the positive x or y direction depending on "phi"

```
x step: if  0<phi<45 or 135<phi<180  
y step: if 45<phi<135
```

For each pixel step in the x,y direction the opposite y,x pixel coordinate is determined by the COF. From our code example the aperture is moved in a x pixel step direction, which means for each step the belonging y coordinate is determine by the COF from the total circular aperture. Just as for elliptical aperture, "q" is here the width of the sky background aperture. The advantage of the trace aperture is, if it turns out that the satellite is very unstable, as long as the Signal to Noise Ratio (SNR) is sufficiently high, this routine will still follow the perhaps strange pattern of the COF for the stars.

The third argument for the utility "aperture_photometry" is if a local or global sky background flux should be used to correct the stellar flux. As mentioned above the local sky background flux is define by a band of width "q" around the stellar aperture. As the factor of stellar contamination and crowding is very hard to predict for our mission, the sky background flux can also be determined globally. This is done simply by slicing the image into s number of subframes. Inside each subframe n number of pixels having the lowest flux is found, hence, s*n is the total number of sky background pixels and the robust 3*median(sky-pixels)-2*mean(sky-pixels) value of the s*n number of pixels with lowest flux is then the sky background flux. When a high level of vignetting or other image artifacts is present the local sky background flux should be used. Also, the global sky background routine do not work for all subframes at the moment. 


# Output



# Usage and dependencies
This software was written October 2018 (Copyright: Nicholas Jannsen) and typeset in Python 3. `blues` has been succesfully tested with python3.5-7 on a Linux system such as Ubuntu 16-20.
   1. numpy
   1. math
   1. scipy
   1. pylab
   1. matplotlib
   1. astropy
   1. PyAstronomy
   1. sys
   1. time
   1. glob
   1. heapq
   1. bottleneck
   1. uncertainties
With the newest installation of `pip`, all of the packages above can be installed using e.g.:

```pip install numpy```

# Technical reports
Aside from the scientific scope of this project some technical issues needed to be looked at and improved. The 'TechnicalReport1*' and 'TechnicalReport2*' is related to my travel the 3. September 2017. 

