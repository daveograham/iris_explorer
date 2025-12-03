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
from explorer.iris_raster_explorer import RasterExplorer

rdir = '/example/path/myrasterfiles'

RasterExplorer(rdir, xpad=8);


# Usage
Choose from the available wavelength windows from the dropdown.

Drag the file slider to scan through the rasters in your directory

The initial pixel intensity values are the mean over the entire available wavelength range.

Clicking the raster window will show the selected pixel spectra in the panel below

The selected pixel is now held if you scrub through file slider

Clicking on the spectrum will display an image at that wavelength bin in the raster window above

Clicking and dragging a portion of the spectrum will produce a raster image of mean intensity over the selected wavelength range.

Choosing another raster window will reset the pixel and wavelength selection

# Fitting (experimental)
Perform a quick fit by pressing Save start parameters

Click the spectrum where you want the peaks

Press test fit

(will at a later date hold the fit it will update when the timeline is scrubbed)

# Comments
Enjoy! Please get in touch with suggestions and or bugs
