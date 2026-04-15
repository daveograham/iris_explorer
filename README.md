# iris_explorer
An ipywidget for exploring IRIS Spectrograph data cubes

At the moment this only works inside a jupyter notebook. It requires ipywidgets to be installed in your environment.

Required arguments:
A path to a directory of IRIS raster files

Optional arguments:
Starting raster file number
xpad - Padding multiplier on the slit scan axis to image with a more helpful aspect ratio (Default = 1)
gui_scale - Integer to scale the overall GUI size (Default = 8)

An example call looks like this:

%matplotlib widget
from iris_explorer import explorer

rdir = '/example/path/myrasterfiles'
explorer.UI(rdir, xpad=8);


# Usage - Rasters
Choose from the available wavelength windows from the dropdown.

Drag the file slider to scan through the rasters in your directory

The image shown is the mean intensity value over the entire window wavelength range.

Clicking the raster window will show the selected pixel spectra in the spectrum panel below

The selected pixel is now held if you scrub the file slider

Clicking on the spectrum will display an image at that wavelength bin in the raster window above

Clicking and dragging a portion of the spectrum will produce a raster image of mean intensity over the selected wavelength range.

Choosing another raster window will reset the pixel and wavelength selection

# Fitting (experimental)
Start a quick fit by pressing Save start parameters

Click on the spectrum where you want the peak starting parameters

Press test fit

Once you have a test fit press Hold Fit and you can now move to another pixel and fit with the same fit setup. You may also move the time range and it'll hold the current position.

# Sit and Stare Mode - with a warning!!!

This is new, and comes with a warning. These files can be huge and at the moment this code will read the whole data cube for all the wavelength channels at once. This is not optimized and can use almost 32gb ram so be careful!

Make sure to add the sitandstare=True flag to your call

This time it takes a single .fits file for the observation. You now have a time range adjustment and scan the pixel through each exposure at the top.

The fitting works in the same manner as the raster.


# Comments
Enjoy! Please get in touch with suggestions and or bugs
