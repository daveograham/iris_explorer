import os
import warnings
import IPython.display as display
from ipywidgets import widgets
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.backend_bases import MouseEvent

from irispy.io import read_files

import numpy as np

class RasterExplorer:
    '''
    Class for selecting rasters and centroids for fitting. Setup mostly borrowed from Chris Osborne's Sunspot Selector
    and heavily inspired by the SSW routine iris_raster_browser
    
    Parameters
    ----------
    fileDir : Str - Directory of raster files
    iraster : Int - Starting Raster file list index (default = 0)
    xpad : Int - integer to rebin the raster step axis by in the plots (default = 1)
    gui_scale : Int - integer to scale the GUI size by (default = 8)
    ----------
    
    '''
    def __init__(self, filedir, iraster = 0, xpad = 1, gui_scale = 8):
        plt.close('all')
        self.filedir = filedir
        
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
        self.wavslice = -1
        self.intset = -1

        raster_obj = read_files(startpath, memmap=False)
        keys = raster_obj.keys()
        self.keylist = list(keys)
        self.rasterkey = self.keylist[0]
        
        #default cursor position
        shape = raster_obj[self.keylist[0]][0].data.shape
        self.xy = (shape[0] // 2, shape[1] // 2)

        self.fig, self.ax = plt.subplot_mosaic('''AB
                                               AB
                                               CC
                                               ''',figsize=(self.scalex,self.scaley))

        self.fig.canvas.header_visible = False
        #dict to store current window state keys        
        self.state = {'A':self.rasterkey,
                      'B':self.keylist[-1]}

        self.setup_buttons(iraster)

        self.load_raster(iraster)
        self.plot_raster_window(self.rasterkey,'A')
        self.plot_raster_window(self.keylist[-1],'B')
        self.plot_spectrum(self.rasterkey, self.xy)

    def setup_buttons(self, startingSlider=0):
        self.receiver = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.receiver = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.slider = widgets.IntSlider(startingSlider, 0, len(self.filelist)-1, description='File Number')
        display.display(self.slider)
        self.slider.observe(self.change_raster, names='value')

        self.window_picker = widgets.Dropdown(options=self.keylist, value=self.rasterkey, description='Window:', disabled=False)
        display.display(self.window_picker)
        self.window_picker.observe(self.change_window, names='value')
        
        self.intslider = widgets.IntRangeSlider(value=[0, 1000],
                                min=0,
                                max=1000,
                                step=10,
                                description='Intensity Scale:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d',
                            )
        display.display(self.intslider)
        self.intslider.observe(self.update_raster, names='value')
        
    def load_raster(self, rx):
        '''open raster file into memory'''
        self.filepath = self.filedir + self.filelist[rx]
        self.raster_obj = read_files(self.filepath, memmap=False)

    def plot_raster_window(self, windowkey, subplot, wavpix=None, drag=0, intlimits=-1):
        raster = self.raster_obj[windowkey][0]
        ldim = raster.data.shape[2]
        #clip to sensible range and mean over wavelength
        #DEFAULT START PLOT -
        if wavpix is None:
            clipped = np.mean(np.clip(raster.data,0,100000),axis=2)
        
        #WHEN CLICKED
        if wavpix is not None:
            if drag == 0:
                clipped = np.clip(raster[:,:,wavpix].data,0,100000)
            else:
                lowx = np.clip(np.min([self.wavpix_down, self.wavpix_up]), 0, ldim)
                highx = np.clip(np.max([self.wavpix_down,self.wavpix_up]), 0, ldim)
                self.wavslice = (lowx,highx)
                clipped = np.mean(np.clip(raster[:,:,lowx:highx].data,0,100000),axis=2)
        
        if intlimits == -1:
            self.ax[subplot].imshow(clipped.T, origin='lower', aspect=1/self.stretch, interpolation='none')
        else:
            self.ax[subplot].imshow(clipped.T, origin='lower', aspect=1/self.stretch, interpolation='none', vmin=intlimits[0], vmax=intlimits[1])
        self.ax[subplot].set_title(windowkey)

        self.state[subplot] = windowkey

    def get_wave(self, windowkey):
        #ignore annoying astropy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.raster_obj[windowkey][0].axis_world_coords("em.wl")[0].to("AA")

    def plot_spectrum(self, windowkey, dataloc, click=-1):
        raster = self.raster_obj[windowkey][0]
        wave = self.get_wave(windowkey)
        x = dataloc[0]
        y = dataloc[1]
        self.ax['C'].clear()
        
        self.ax['C'].set_ylim([0, 1.2*np.max(raster[x,y].data)])
        self.ax['C'].step(wave, raster[x,y].data)
        
        if self.markerA is not None:
            self.markerA[0].remove()
            self.markerB[0].remove()
        
        self.markerA = self.ax['A'].plot(x,y, marker='+',color='w')  #markersize
        self.markerB = self.ax['B'].plot(x,y, marker='+',color='w')
        
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
            self.plot_spectrum(self.rasterkey, self.xy)
            self.axclicked = event.inaxes
            if self.wavslice != -1:
                self.plot_slice()

        #IF A SPECTRUM CLICK ==========================
        if event.inaxes is self.ax['C']:
            self.wavloc_down = (event.xdata, event.ydata)
            #get current clicked (left for now) window key
            wave = self.get_wave(self.rasterkey)
            #mean wavelength pixel width
            self.dwave = abs(np.mean(wave.value[:-1]-wave.value[1:]))
            #interpolate to get clicked pixel number from wavelength value
            self.wavpix_down = np.searchsorted(wave.value, self.wavloc_down[0])
            
            #self.plot_raster_window(self.rasterkey,'A', self.wavpix_down)
            self.plot_spectrum(self.rasterkey, self.xy, click=1)
            
    def on_release(self, event: MouseEvent):
        if self.fig.canvas.manager.toolbar.mode != '':
            return
        if event.inaxes is self.ax['C']:
            self.wavloc_up = (event.xdata, event.ydata)
            clickdiff = abs(event.xdata-self.wavloc_down[0])

            wave = self.get_wave(self.rasterkey)
            self.wavpix_up = np.searchsorted(wave.value, self.wavloc_up[0])

            if self.wavloc_down[0] == event.xdata:
                self.plot_raster_window(self.rasterkey,'A', self.wavpix_down)
                return

            if clickdiff < self.dwave*500.0 and clickdiff > self.dwave:  #check clicks are more than 1 pixel away
                self.ax['C'].plot((event.xdata,event.xdata),(0,100000),color='g',linestyle='--')
                #interpolate to get clicked pixel number from wavelength value
                self.plot_raster_window(self.rasterkey,'A', self.wavpix_up, drag=1)
                
                self.shade_spectrum(wave)
            else:
                self.plot_raster_window(self.rasterkey,'A', self.wavpix_down)

    def shade_spectrum(self, wave):
        self.ax['C'].fill_between(wave[self.wavpix_down:self.wavpix_up].value, 0, 100000, facecolor='green',alpha=0.25)

    def plot_slice(self):
        wave = self.get_wave(self.rasterkey)
        wlow = wave[self.wavslice[0]].value
        whigh = wave[self.wavslice[1]].value
        self.shade_spectrum(wave)
        self.ax['C'].plot((wlow,wlow),(0,100000),color='r',linestyle='--')
        self.ax['C'].plot((whigh,whigh),(0,100000),color='g',linestyle='--')


    #UI DRIVERS =================================================
    def change_window(self, event):
        self.plot_raster_window(event['new'],'A')
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
        self.plot_raster_window(self.rasterkey,'A', intlimits=self.intset)
        self.plot_raster_window('Mg II k 2796','B')
        #update spectrum and markers on raster
        if hasattr(self, 'xy'):
            self.plot_spectrum(self.rasterkey, self.xy)
        if self.wavslice != -1:   #hasattr(self, 'wavslice'):
            self.plot_slice()

    def update_raster(self, event):
        #update raster intensity scale
        #self.intevent = event
        if self.wavslice != - 1:
            self.plot_raster_window(self.rasterkey,'A',intlimits=event['new'],drag=1)
        else:
            self.plot_raster_window(self.rasterkey,'A',intlimits=event['new'],drag=0)

        self.intset = event['new']

    def disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def clear(self, _):
        self.hmiCoords[:] = []
        self.boxCoords[:] = []
        for pIdx in range(len(self.ax.patches)-1, -1, -1):
            self.ax.patches[pIdx].remove()
        self.fig.canvas.flush_events()
        self.fig.canvas.flush_events()