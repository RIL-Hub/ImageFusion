import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tifffile import imread
import glob
import pydicom as dicom
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ImageView:
    def __init__(self, parent, X, view):
        self.fig = plt.Figure()
        self.fig.set_figheight(3)
        self.fig.set_figwidth(3)
        self.ax = self.fig.add_subplot()
        
        self.view = view
        self.X = X
        self.slice = self.X.get_slice(view, 99, 'by_number')
        self.intensity_limits = [0.0, self.X.max_intensity]
        
        self.image = self.ax.imshow(self.slice, vmin=0, vmax=self.X.max_intensity, cmap='gist_yarg', interpolation='none')
        
        # Color Bar
        divider_PET = make_axes_locatable(self.ax)
        cb_cax = divider_PET.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(self.image, cax=cb_cax)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(padx=2.5, pady=2.5, expand=1, fill='both')
    
    def set_slice(self, slice_indicator, mode):
        self.slice = self.X.get_slice(self.view, slice_indicator, mode)
        self.update_data()
        
    def set_intensity(self, intensity_limits):
        self.intensity_limits = [intensity_limits[0]*self.X.max_intensity, intensity_limits[1]*self.X.max_intensity]
        self.update_data()
    
    def set_cmap(self, cmap):
        self.image.set_cmap(cmap)
        self.update_data()
    
    def update_data(self):
        self.image.set_data(np.clip(self.slice, self.intensity_limits[0], self.intensity_limits[1]))
        self.image.set_clim(vmin=self.intensity_limits[0], vmax=self.intensity_limits[1])
        self.cbar.update_normal(self.image)
        
        # self.canvas.flush_events()
        self.canvas.draw()

class DualImageView:
    def __init__(self, parent, image_1_view, image_2_view):
        self.opacity = 0.5
        
        self.fig = plt.Figure()
        self.fig.set_figheight(3)
        self.fig.set_figwidth(3)
        self.ax = self.fig.add_subplot()
        
        self.image_1_view = image_1_view
        self.image_2_view = image_2_view
        
        self.image_1 = self.ax.imshow(self.image_1_view.slice, cmap='gist_gray', interpolation='none')
        self.image_2 = self.ax.imshow(self.image_2_view.slice, cmap='magma', alpha=self.opacity, interpolation='none', extent=self.image_1.get_extent())
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(padx=2.5, pady=2.5, expand=1, fill='both')

    def set_slice(self, slice_indicator, mode):
        # rely on base image views being set
        self.update_data()
        
    def set_intensity(self, intensity_limits):
        # TODO: make second slider for each image?
        ...
    
    def update_data(self):
        self.image_1.set_data(self.image_1_view.slice)
        self.image_2.set_data(self.image_2_view.slice)
        # self.cbar.update_normal(self.image_1)
        # self.cbar.update_normal(self.image_2)
        
        # self.canvas.flush_events()
        self.canvas.draw()

class ImageData:
    def __init__(self, file_properties, image_type):
        self.file_properties = file_properties
        self.image_type = image_type
        
        if self.image_type == 'PET':
            self.init_PET_image()
        if self.image_type == 'CT':
            self.init_CT_image()
        if self.image_type == 'dual':
            self.init_dual_image()
        
        self.X = self.original_X.copy()
        self.max_intensity = np.amax(self.X)
        
        self.print_info()
       
    def print_info(self):
        print(f'{self.image_type}')
        print(f'Image Size: {self.vxls_in_dim[0]} x {self.vxls_in_dim[1]} x {self.vxls_in_dim[2]} vxls')
        print(f'Image Dims: [{self.dims[0]}, {self.dims[1]}, {self.dims[2]}] mm')
        print(f'Voxel Dims: [{self.vxl_dims[0]}, {self.vxl_dims[1]}, {self.vxl_dims[2]}] mm')
        print('')

    def init_PET_image(self):
        self.original_X = np.flip(imread(self.file_properties['path']), axis=0)
        
        vxl_dim1 = self.file_properties['vxl_dim1_size']
        vxl_dim2 = self.file_properties['vxl_dim2_size']
        vxl_dim3 = self.file_properties['vxl_dim3_size']
        self.vxl_dims = [vxl_dim1, vxl_dim2, vxl_dim3]
        
        vxls_in_dim1 = int(self.original_X.shape[0])
        vxls_in_dim2 = int(self.original_X.shape[1])
        vxls_in_dim3 = int(self.original_X.shape[2])
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        
        self.set_dims()
        
    def init_CT_image(self):
        plots = []
        file_names = []
        vxls_in_dim3 = 0
        for idx, file_name in enumerate(glob.glob(self.file_properties['path'] + '*.IMA')):
            ds = dicom.dcmread(file_name)
            pix = ds.pixel_array
            plots.append(pix)
            file_name = file_name.split('\\')[-1]
            file_names.append(file_name)  # Save the filename
            vxls_in_dim3 = vxls_in_dim3 + 1
            if idx == 0:
                vxl_dim1 = float(ds.PixelSpacing[0])
                vxl_dim2 = float(ds.PixelSpacing[1])
                vxl_dim3 = float(ds.SliceThickness)
                vxls_in_dim1 = int(ds.Rows)
                vxls_in_dim2 = int(ds.Columns)
        
        self.vxl_dims = [vxl_dim1, vxl_dim2, vxl_dim3]
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        self.original_X = np.dstack(plots)
        self.set_dims()
    
    def init_dual_image(self):
        self.vxl_dims = [1, 1, 1]
        self.vxls_in_dim = [100, 100, 100]
        self.original_X = np.zeros([100,100,100])
        self.set_dims()
    
    def get_slice_by_number(self, slice_number, view):
        slice_number = int(slice_number)
        if view == 0: return self.X[slice_number, :, :]
        if view == 1: return self.X[:, slice_number, :]
        if view == 2: return self.X[:, :, slice_number]
    
    def get_slice(self, view, slice_indicator, mode):
        if mode == 'by_number':
            slice_number = slice_indicator
        if mode == 'by_percent':
            slice_number = int(np.max([0, np.floor(slice_indicator * (self.vxls_in_dim[view])-1)]))
        if view == 0: return self.X[slice_number, :, :]
        if view == 1: return self.X[:, slice_number, :]
        if view == 2: return self.X[:, :, slice_number]
    
    def pad_to_dims(self, target_dims):
        for i, (X_dim, T_dim) in enumerate(zip(self.dims, target_dims)):
            if X_dim < T_dim:
                voxels_to_add = int(round( (T_dim - X_dim)/self.vxl_dims[i] ))
                padding_width = [[0, 0], [0, 0], [0, 0]]
                padding_width[i][0] = voxels_to_add // 2
                padding_width[i][1] = voxels_to_add // 2
                self.X = np.pad(self.X, padding_width)
            
        self.set_vxls_in_dim()
    
    def set_vxls_in_dim(self):
        vxls_in_dim1 = int(self.X.shape[0])
        vxls_in_dim2 = int(self.X.shape[1])
        vxls_in_dim3 = int(self.X.shape[2])
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        self.set_dims()
    
    def set_dims(self):
        self.dims = np.multiply(self.vxl_dims, self.vxls_in_dim)