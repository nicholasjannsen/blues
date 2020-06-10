# A BlueSONG: Tracing magnetic activity in the Ca II lines of solar-type stars using eShel, a commercial spectrograph mounted on the Hertzsprung SONG telescope
This repository holds my Master's thesis and all the codes related to it. Following the project abstract a description of the software and the some technical aspects are given. 

# Project abstract
This project involves the commercial échelle spectrograph, eShel, mounted on the fully robotic Hertzsprung SONG telescope on Tenerife, to prove the concept of using an affordable spectrograph, to trace stellar magnetic activity in the Ca II lines of solar-type stars. The eShel is a fiber-fed échelle spectrograph with a spectral resolving power of R ~ 10,000, and it has proven to be a useful tool for the detection of radial velocities down to 80 m/s, but with the difficulty of receiving a sufficient emission excess in the near-UV, this project is to our knowledge the first of its kind. Since the strongest spectral features observable from ground are the Ca II lines, and the fact that long-term observations of stellar chromospheric activity primarily comes from the Mt Wilson 30-year HK-Project, this research likewise use the Ca II lines as a proxy of stellar magnetic activity. The project objectives are thus; first to investigate that an off-the-shelf spectrograph like eShel can provide sufficient Ca II line emission by observations. Secondly, to calculate the chromospheric activity proxy $S$ and, thirdly, evaluate the success off monitoring activity cycles of stars similar to the Sun using eShel. To meet these requirements, observations obtained with eShel, were supplemented by simultaneous observations made with the FIES spectrograph at the Nordic Optical Telescope, and both dataset was directly compared to the Mt Wilson survey. In a near-future with an increasing number of larger ground and space based telescopes, such as the TMT, ELT, PLATO, JWST, etc., an increase of small and fully robotic telescopes will be needed to complement monitoring, survey, and follow-up observations, of which we begin to see their infancy. Hence, the fact that this master's thesis demonstrates, that the eShel spectrograph can be used to successfully monitor stellar magnetic activity of solar-type stars, confirms that commercial spectrographs may be of significant importance, for future networks of large surveys within the scope of observing and understanding stellar activity in solar-type stars. If proving worthy in terms of efficiency, eShel will provide a possibility and great potential for observing stellar cycles using the Hertzsprung SONG telescope. 

# Software description: `blues.py`
This python class is specifically made for the spectroscopic data reduction of the Shelyak eShel spectrograph
which is installed at the Hertzsprung SONG node telescope at Tenrife, Spain. The software is originally built
from structures of the 'SONGWriter' which is SONG's spectroscopic data reduction pipeline, and by others is 
inspired by the data reduction pipeline 'FIESTools' of the FIES spectrograph at the NOT on La Palma.

A detailed description of blues can be found in my Masterś thesis under the chapter "5. Pipeline: Blues". In short the combined software consist of `blues.py`, being the main script running modules from `BlueSONG-py` and `Plot_Tools.py`, for which the names suggests, are the main librirary of science modules and associated plot modules, respectively.  

# Dependencies
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

# Usage:
The usage is straight forward since only `blues.py` needs to run from a python or ipython shell. However, one need to point the following directories indicated by strings; i) where the data to be analyzed is stored (`path/to/data/`), ii) where the software is stored,  and iii) where images and results produced by the software needs to be stored. The first can be found the `blues.py` and the two latter in `BlueSONG.py`.

Looking into `blues.py` in more detail, it is here possible to skip the first step, the image reduction, (use `skip=0`) if the data already have been reduced by another software or have already been analyzed with blues. Also each module has a illustrative plotting module in order to visulaize the results, which can be activated using `plot=1`. This helps you both to understand the software and for troubleshooting. 

# Output
Ultimately the output of blues is a fully reduced spectrum around the Ca II lines and a prompt to the terminal of the S index activity proxy calculated using different spectral band definitions.  

# Technical reports
Aside the eShel manual (named "eShel_Installation_and_Maintenance_Manual.pdf" in this repository) from the scientific scope of this project some technical issues needed to be looked at and improved. The 'Technical_Reports.pdf' is attached to this repository where all the technical work and complications with eShel are found. 

