import os
import warnings
import IPython.display as display
from ipywidgets import widgets
import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseEvent

from irispy.io import read_files

import numpy as np

from enum import Enum

from scipy.optimize import curve_fit

from .fit_tools import ngen_gauss
from .fit_tools import gaussian_bounds
# version to work with a sit and stare

class UI:
    '''
    Class for selecting rasters and centroids for fitting. Setup mostly borrowed from Chris Osborne's Sunspot Selector
    and inspired by the SSW routine iris_raster_browser
    
    Parameters
    ----------
    fileDir : Str - Directory of raster files
    iraster : Int - Starting Raster file list index (default = 0)
    xpad : Int - integer to rebin the raster step axis by in the plots (default = 1)
    gui_scale : Int - integer to scale the GUI size by (default = 8)
    ----------
    
    '''
    class Mode(Enum):
        PIXEL = "pixel"
        FIT = "fit"
        HOLD = "hold"

    
    def __init__(self, filedir, iraster = 0, xpad = 1, gui_scale = 8, memsave = False, sitandstare = False):
        plt.close('all')
        self.filedir = filedir
        self.memsave = memsave
        self.sitandstare = sitandstare
        self.filelist = os.listdir(self.filedir)
        
        #some setup
        startpath = filedir + self.filelist[iraster]
        self.stretch = xpad

        self.markerA = None
        self.markerB = None

        #set GUI size
        self.scalex = gui_scale
        self.scaley = gui_scale

        #default states
        self.mode = self.Mode.PIXEL
        self.wavslice = -1
        self.intset = [0, 50] #default intensity value in % of max (0-100%)
        self.timeset = [0, 1] #default fraction of time range to use for SNS x axis (odd behaviour when not set to 0 1)

        #fitter default width
        self.pwid = 0.1

        if memsave is True:
            raster_obj = read_files(startpath, memmap=True)
        else:
            raster_obj = read_files(startpath, memmap=False)

        keys = raster_obj.keys()
        self.keylist = list(keys)
        self.rasterkey = self.keylist[0]
        
        #default cursor position
        idims = raster_obj[self.keylist[0]][0].data.shape
        self.idims = idims
        self.xy = (idims[0] // 2, idims[1] // 2)

        #default time range
        if sitandstare == True:
            self.timelimits = [0, int(idims[0]*self.timeset[1]-1)]
        else:
            self.timelimits = [0, int(idims[0]*self.timeset[1]-1)]


        self.fig, self.ax = plt.subplot_mosaic('''AB
                                               AB
                                               CC
                                               ''',figsize=(self.scalex,self.scaley))

        self.fig.canvas.header_visible = False
        #dict to store current window state keys        
        self.state = {'A':self.rasterkey,
                      'B':self.keylist[-1]}

        self.load_raster(iraster)
        self.setup_buttons(iraster)

        if sitandstare == False:
            self.plot_raster_window(self.rasterkey,'A')
            self.plot_raster_window(self.keylist[-1],'B')
        if sitandstare == True:
            self.plot_raster_window(self.rasterkey,'A', timelimits=self.timelimits, intset=self.intset, drag=0)
            self.plot_raster_window(self.keylist[-1],'B', timelimits=self.timelimits, intset=self.intset, drag=0)

        self.plot_spectrum(self.rasterkey, self.xy)


    def setup_buttons(self, startingSlider=0):
        #BASIC CLICKS
        self.receiver = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.receiver = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        #RASTER SLIDER
        if self.sitandstare == False:
            self.expslider = widgets.IntSlider(startingSlider, 0, len(self.filelist)-1, description='File Number')
            display.display(self.expslider)
            self.expslider.observe(self.change_raster, names='value')
        else:
            self.expslider = widgets.IntSlider(startingSlider, 0, self.idims[0]-1, description='Exposure')
            display.display(self.expslider)
            self.expslider.observe(self.change_exposure, names='value')

        #WINDOW SELECTION
        self.window_picker = widgets.Dropdown(options=self.keylist, value=self.rasterkey, description='Window:', disabled=False)
        display.display(self.window_picker)
        self.window_picker.observe(self.change_window, names='value')

        #INTENSITY SLIDER        
        self.intslider = widgets.IntRangeSlider(value=self.intset,
                                min=0,
                                max=100,
                                step=1,
                                description='Intensity Scale:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d',
                            )
        display.display(self.intslider)
        self.intslider.observe(self.update_raster, names='value')

        if self.sitandstare == True:
            #SNS CLIP SLIDER
            self.timeslider = widgets.IntRangeSlider(value=[0, int(self.timeset[1]*self.idims[0]-1)],
                                    min=0,
                                    max=self.idims[0]*self.timeset[1]-1,
                                    step=10,
                                    description='Time Range:',
                                    disabled=False,
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='d',
                                )
            display.display(self.timeslider)
            self.timeslider.observe(self.update_timeslice, names='value')

        #START PARAMETER PICKER
        self.paramstartButton = widgets.ToggleButton(
                value=False,
                description='Save start parameters',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Choose starting centroid and intensity for gaussian fitting',
                icon='' # (FontAwesome names without the `fa-` prefix)
        )
        display.display(self.paramstartButton)
        self.paramstartButton.observe(self.param_starter, names='value')

        self.runfitButton = widgets.ToggleButton(
        value=False,
        description='Test fit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Run a fit based on the current start parameters and pixel',
        icon='' # (FontAwesome names without the `fa-` prefix)
        )
        display.display(self.runfitButton)
        self.runfitButton.observe(self.run_fit, names='value')

        self.holdfitButton = widgets.ToggleButton(
        value=False,
        description='Hold fit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Hold current fit parameters',
        icon='' # (FontAwesome names without the `fa-` prefix)
        )
        display.display(self.holdfitButton)
        self.holdfitButton.observe(self.hold_fit, names='value')

    def load_raster(self, rx):
        '''open raster file into memory'''
        self.filepath = self.filedir + self.filelist[rx]
        if self.memsave is True:
            self.raster_obj = read_files(self.filepath, memmap=True)
        else:
            self.raster_obj = read_files(self.filepath, memmap=False)

    def plot_raster_window(self, windowkey, subplot, wavpix=None, drag=0, intset=-1, timelimits=-1):
        raster = self.raster_obj[windowkey][0]
        ldim = raster.data.shape[2]
        xdim = raster.data.shape[0]

        if timelimits == -1:
            timelimits = [0,xdim-1]

        #clip to sensible range and mean over wavelength
        #DEFAULT START PLOT -
        if wavpix is None:
            clipped = np.mean(np.clip(raster.data[timelimits[0]:timelimits[1],:,:],0,100000),axis=2)

        if intset == -1:
            intset = [0, 50]

        #WHEN CLICKED
        if wavpix is not None:
            if drag == 0:
                clipped = np.clip(raster[timelimits[0]:timelimits[1],:,wavpix].data,0,100000)
            else:
                lowx = np.clip(np.min([self.wavpix_down, self.wavpix_up]), 0, ldim)
                highx = np.clip(np.max([self.wavpix_down, self.wavpix_up]), 0, ldim)
                self.wavslice = (lowx,highx)
                clipped = np.mean(np.clip(raster[timelimits[0]:timelimits[1],:,lowx:highx].data,0,100000),axis=2)
        
        if timelimits != -1:
            self.ax[subplot].imshow(clipped.T, origin='lower', aspect='auto', interpolation='none', vmin=np.max(clipped)*intset[0]*0.01, vmax=np.max(clipped)*intset[1]*0.01)
        else:
            self.ax[subplot].imshow(clipped.T, origin='lower', aspect='auto', interpolation='none', vmin=np.max(clipped)*intset[0]*0.01, vmax=np.max(clipped)*intset[1]*0.01)
        
        self.ax[subplot].set_title(windowkey)

        self.state[subplot] = windowkey
        self.timelimits = timelimits

    def get_wave(self, windowkey):
        #ignore annoying astropy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.raster_obj[windowkey][0].axis_world_coords("em.wl")[0].to("AA")

    def plot_spectrum(self, windowkey, plotloc, click=-1):
        raster = self.raster_obj[windowkey][0]
        wave = self.get_wave(windowkey)

        xdata = plotloc[0] + self.timelimits[0]
        ydata = plotloc[1]

        self.ax['C'].clear()
        
        self.ax['C'].set_ylim([0, 1.2*np.max(raster[xdata,ydata].data)])
        self.ax['C'].step(wave, raster[xdata ,ydata].data)
        
        if self.markerA is not None:
            self.markerA[0].remove()
            self.markerB[0].remove()
        
        self.markerA = self.ax['A'].plot(plotloc[0],plotloc[1], marker='+',color='r', scalex=False, scaley=False, zorder=10)  #markersize
        self.markerB = self.ax['B'].plot(plotloc[0],plotloc[1], marker='+',color='r', scalex=False, scaley=False, zorder=10)
        
        #plot marker for selection
        if click > 0:
            self.ax['C'].plot((self.wavloc_down[0],self.wavloc_down[0]),(0,100000),color='r',linestyle='--')

    def on_click(self, event: MouseEvent):
        if self.fig.canvas.manager.toolbar.mode != '':
            return
        if event.xdata is None:
            return
        
        pixloc = (int(event.xdata+0.5), int(event.ydata+0.5))
        
        #IF A RASTER CLICK ============================        
        if event.inaxes is self.ax['A'] or event.inaxes is self.ax['B']:
            self.xy = pixloc
            if self.sitandstare == True:
                self.expslider.value = pixloc[0] - self.timelimits[0]

            self.plot_spectrum(self.rasterkey, self.xy)
            if self.mode is self.Mode.HOLD:
                self.fitter(self.rasterkey, self.xy)
            self.axclicked = event.inaxes
            if self.wavslice != -1:
                self.plot_slice()
            
            #reset parameter picker to zero
            if self.mode == self.Mode.FIT:
                self.mode = self.Mode.PIXEL
                self.paramstartButton.value = False
                self.fitmarkers = []
                self.pstart = []                

        #IF A SPECTRUM CLICK ==========================
        if event.inaxes is self.ax['C']:
            #MODE CHECK what mode we're on
            if self.mode == self.Mode.PIXEL:
                self.wavloc_down = (event.xdata, event.ydata)
                #get current clicked (left for now) window key
                wave = self.get_wave(self.rasterkey)
                #mean wavelength pixel width
                self.dwave = abs(np.mean(wave.value[:-1]-wave.value[1:]))
                #interpolate to get clicked pixel number from wavelength value
                self.wavpix_down = np.searchsorted(wave.value, self.wavloc_down[0])
                
                self.plot_spectrum(self.rasterkey, self.xy, click=1)
            
            if self.mode == self.Mode.FIT or self.mode == self.Mode.HOLD:
                pcen = event.xdata
                pint = event.ydata

                marker = self.ax['C'].plot(pcen, pint, marker='+',color='r')
                self.fitmarkers.append(marker)

                #add the I & centroid start params
                self.gausscount += 1
                self.pstart.append(pint)
                self.pstart.append(pcen)
                self.pstart.append(self.pwid)

    def on_release(self, event: MouseEvent):
        if self.fig.canvas.manager.toolbar.mode != '':
            return
        if event.inaxes is self.ax['C']:
            if self.mode == self.Mode.PIXEL:
                self.wavloc_up = (event.xdata, event.ydata)
                clickdiff = abs(event.xdata-self.wavloc_down[0])

                wave = self.get_wave(self.rasterkey)
                self.wavpix_up = np.searchsorted(wave.value, self.wavloc_up[0])

                if self.wavloc_down[0] == event.xdata:
                    self.plot_raster_window(self.rasterkey,'A', self.wavpix_down, timelimits=self.timelimits, intset=self.intset)
                    return

                if clickdiff < self.dwave*500.0 and clickdiff > self.dwave:  #check clicks are more than 1 pixel away
                    self.ax['C'].plot((event.xdata,event.xdata),(0,100000),color='g',linestyle='--')
                    #interpolate to get clicked pixel number from wavelength value
                    self.plot_raster_window(self.rasterkey,'A', self.wavpix_up, timelimits=self.timelimits, intset=self.intset, drag=1)
                    
                    self.shade_spectrum(wave)
                else:
                    self.plot_raster_window(self.rasterkey,'A', self.wavpix_down, timelimits=self.timelimits, intset=self.intset)

    def shade_spectrum(self, wave):
        self.ax['C'].fill_between(wave[self.wavpix_down:self.wavpix_up].value, 0, 100000, facecolor='green',alpha=0.25)

    def plot_slice(self):
        wave = self.get_wave(self.rasterkey)
        wlow = wave[self.wavslice[0]].value
        whigh = wave[self.wavslice[1]].value
        self.shade_spectrum(wave)
        self.ax['C'].plot((wlow,wlow),(0,100000),color='r',linestyle='--')
        self.ax['C'].plot((whigh,whigh),(0,100000),color='g',linestyle='--')


    def fitter(self, windowkey, dataloc):
        x = dataloc[0]
        y = dataloc[1]

        raster = self.raster_obj[windowkey][0]
        wave = self.get_wave(windowkey)
        spec = raster[x,y].data

        bounds = gaussian_bounds(self.pstart)

        yparam, ycov = curve_fit(ngen_gauss, wave, spec, p0 = self.pstart, bounds=bounds)

        self.yparam = yparam
        fit = ngen_gauss(wave.value, *yparam)
        self.ax['C'].plot(wave, fit, linestyle='--')

        for n in range(0,self.gausscount):
            fit = ngen_gauss(wave.value, yparam[n*3:n*3+3])
            self.ax['C'].plot(wave, fit)


    #UI DRIVERS =================================================
    def run_fit(self, event):
        if len(self.pstart) == 0:
            return
        if self.mode is self.Mode.FIT:
            self.paramstartButton.value = False
            self.fitter(self.rasterkey, self.xy)
            self.runfitButton.value = False
            return
        
    def hold_fit(self, event):
        if event['new'] == True:
            self.mode = self.Mode.HOLD
        if event['new'] == False:
            self.mode = self.Mode.PIXEL
            self.fitmarkers = []
            self.plot_spectrum(self.rasterkey, self.xy)
            return
        
    def param_starter(self, event):
        if event['new'] == True:
            self.mode = self.Mode.FIT
            self.fitmarkers = []
            self.gausscount = 0
            self.pstart = []
            self.plot_spectrum(self.rasterkey, self.xy)

        if event['new'] == False:
            self.mode = self.Mode.PIXEL
            self.fitmarkers = []
            self.plot_spectrum(self.rasterkey, self.xy)
        return

    def change_window(self, event):
        self.plot_raster_window(event['new'],'A', timelimits=self.timelimits)
        self.rasterkey = event['new']
        #clear any wavelength selection here
        self.wavslice = -1
        self.intset = -1
        #plot new spectrum at current pixel position
        if hasattr(self, 'xy'):
            self.plot_spectrum(event['new'], self.xy)

    def change_raster(self, event):
        #go between to pass the event object into my own function
        self.load_raster(event['new'])
        self.plot_raster_window(self.rasterkey,'A', intset=self.intset, timelimits=self.timelimits)
        self.plot_raster_window('Mg II k 2796','B', intset=self.intset, timelimits=self.timelimits)
        #update spectrum and markers on raster
        if hasattr(self, 'xy'):
            self.plot_spectrum(self.rasterkey, self.xy)
        if self.wavslice != -1:   #hasattr(self, 'wavslice'):
            self.plot_slice()
        if self.mode is self.Mode.HOLD:
            self.fitter(self.rasterkey, self.xy)

    def change_exposure(self, event):
        #update plots for new sit and stare exposure selection

        if hasattr(self, 'xy'):
            #make new data coord for time and y slit pos
            self.plot_spectrum(self.rasterkey, [event['new'], self.xy[1]])
        if self.wavslice != -1:   #hasattr(self, 'wavslice'):
            self.plot_slice()
        if self.mode is self.Mode.HOLD:
            self.fitter(self.rasterkey, [event['new'],self.xy[1]])

    def update_raster(self, event):
        #update raster intensity scale - takes percentage of maximum as the scale
        self.intset = event['new']

        if self.wavslice != - 1:
            self.plot_raster_window(self.rasterkey,'A',intset=event['new'],timelimits=self.timelimits, drag=1)
        else:
            self.plot_raster_window(self.rasterkey,'A',intset=event['new'],timelimits=self.timelimits, drag=0)

        self.intset = event['new']

    def update_timeslice(self, event):
        self.timelimits = event['new']
        self.plot_raster_window(self.rasterkey,'A', intset=self.intset, timelimits = event['new'], drag=0)
        self.plot_raster_window(self.keylist[-1],'B', intset=self.intset, timelimits = event['new'], drag=0)

        if hasattr(self, 'xy'):
            #make new data coord for time and y slit pos
            self.plot_spectrum(self.rasterkey, [self.xy[0]-self.timelimits[0], self.xy[1]])

    def disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def clear(self, _):
        self.hmiCoords[:] = []
        self.boxCoords[:] = []
        for pIdx in range(len(self.ax.patches)-1, -1, -1):
            self.ax.patches[pIdx].remove()
        self.fig.canvas.flush_events()
        self.fig.canvas.flush_events()