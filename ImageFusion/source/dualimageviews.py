import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tifffile import imread
import glob
import pydicom as dicom
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DualImageView:
    def __init__(self, parent, ImageView1, ImageView2):
        self.opacity = 0.5
        
        self.ImageView1 = ImageView1
        self.ImageView2 = ImageView2
        
        self.fig = plt.Figure()
        self.fig.set_figheight(3)
        self.fig.set_figwidth(3)
        self.ax = self.fig.add_subplot()

        self.slice_1 = self.ImageView1.slice
        self.slice_2 = self.ImageView2.slice
        
        self.image_1 = self.ax.imshow(self.slice_1, cmap=self.ImageView1.cmap, interpolation='none')
        self.image_2 = self.ax.imshow(self.slice_2, cmap=self.ImageView2.cmap, alpha=self.opacity, interpolation='none', extent=self.image_1.get_extent())
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')
        
    
    def update_data(self):
        self.slice_1 = self.ImageView1.slice
        self.slice_2 = self.ImageView2.slice
        
        self.image_1.set_cmap(self.ImageView1.cmap)
        self.image_2.set_cmap(self.ImageView2.cmap)
        
        self.image_1.set_data(self.slice_1)
        self.image_2.set_data(self.slice_2)
       
        self.image_1.set_clim(vmin=self.ImageView1.intensity_limits[0], vmax=self.ImageView1.intensity_limits[1])
        self.image_2.set_clim(vmin=self.ImageView2.intensity_limits[0], vmax=self.ImageView2.intensity_limits[1])
        
        # self.canvas.flush_events()
        self.canvas.draw()

# class DualImageView:
#     def __init__(self, parent, X_dual, view):
#         self.opacity = 0.5
        
#         self.fig = plt.Figure()
#         self.fig.set_figheight(3)
#         self.fig.set_figwidth(3)
#         self.ax = self.fig.add_subplot()
        
#         self.view = view
#         self.X_dual = X_dual
#         self.slice_1, self.slice_2 = self.X_dual.get_slice(view, 99, 'by_number')
#         self.intensity_limits_1 = [0.0, self.X_dual.max_intensity_1]
#         self.intensity_limits_2 = [0.0, self.X_dual.max_intensity_2]
        
#         self.image_1 = self.ax.imshow(self.slice_1, cmap='gist_gray', interpolation='none')
#         self.image_2 = self.ax.imshow(self.slice_2, cmap='hot', alpha=self.opacity, interpolation='none', extent=self.image_1.get_extent())
        
#         self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
#         self.canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')

#     def set_slice(self, slice_indicator, mode):
#         # rely on base image views being set
#         self.slice_1, self.slice_2 = self.X_dual.get_slice(self.view, slice_indicator, mode)
#         self.update_data()
        
#     def set_intensity(self, intensity_limits):
#         # TODO: make second slider for each image?
#         ...
    
#     def update_data(self):
#         self.image_1.set_data(self.slice_1)
#         self.image_2.set_data(self.slice_2)
#         # self.cbar.update_normal(self.image_1)
#         # self.cbar.update_normal(self.image_2)
        
#         # self.canvas.flush_events()
#         self.canvas.draw()

class DualImageData:
    def __init__(self, X_1, X_2):
        self.X_1 = X_1
        self.X_2 = X_2
        
        self.set_vxls_in_dim()
        self.max_intensity_1 = np.amax(self.X_1)
        self.max_intensity_2 = np.amax(self.X_2)
    
    def get_slice(self, view, slice_indicator, mode):
        if mode == 'by_number':
            slice_number_1 = slice_indicator
            slice_percent = slice_indicator / (self.X_1.vxls_in_dim[view]-1)
            slice_number_2 = int(np.max([0, np.floor(slice_percent * (self.X_2.vxls_in_dim[view])-1)]))
        if mode == 'by_percent':
            slice_number_1 = int(np.max([0, np.floor(slice_indicator * (self.X_1.vxls_in_dim[view])-1)]))
            slice_number_2 = int(np.max([0, np.floor(slice_indicator * (self.X_2.vxls_in_dim[view])-1)]))
        return self.X_1.get_slice(view, slice_number_1, 'by_number'), self.X_2.get_slice(view, slice_number_2, 'by_number')
        
    def set_vxls_in_dim(self):
        vxls_in_dim1 = int(np.max([self.X_1.vxls_in_dim[0], self.X_2.vxls_in_dim[0]]))
        vxls_in_dim2 = int(np.max([self.X_1.vxls_in_dim[1], self.X_2.vxls_in_dim[1]]))
        vxls_in_dim3 = int(np.max([self.X_1.vxls_in_dim[2], self.X_2.vxls_in_dim[2]]))
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]