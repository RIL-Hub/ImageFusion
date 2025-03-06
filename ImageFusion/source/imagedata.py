import numpy as np
from tifffile import imread
import glob
import pydicom as dicom
import os
import datetime

class ImageData:
    
    def __init__(self, file_properties):
        
        self.dcm = None
        self.file_properties = file_properties
        
        # --- Initialize Image --- #
        
        if os.path.isfile(self.file_properties['path']):
            self.init_file_image()
        elif os.path.isdir(self.file_properties['path']):
            self.init_dir_image()
        else:
            raise RuntimeError("Provided image file path does not exist!")
        
        # --- Initialize Data Structures --- #
        
        # Set Values
        if not isinstance(self.original_X[0,0,0], np.uint16):
            necessary_type = np.uint16
            self.X = self.original_X.copy() - np.min(self.original_X)
            max_value = np.max(self.X)
            self.X = (self.X / max_value) * np.iinfo(necessary_type).max
            self.intercept = np.min(self.original_X) 
            self.slope = max_value / np.iinfo(necessary_type).max
            self.X = self.X.astype(necessary_type)
        else:
            self.X = self.original_X.copy()
        
        self.refresh_characteristics()
        self.max_intensity = np.amax(self.X)
        self.slice_index = [0, 0, 0]
        self.slice_mm = [0, 0, 0]
        self.preload_slices()
        
        # --- Create DICOM stucture --- #
        
        self.print_info()
        if not self.dcm:
            self.create_dcm_object()
        self.update_dcm_object()
       
    # --- Image Initialization --- #
    
    def print_info(self):
        print(f'{self.file_properties["path"]}')
        print(f'Image Size: {self.vxls_per_view[0]} x {self.vxls_per_view[1]} x {self.vxls_per_view[2]} vxls')
        print(f'Image Dims: [{self.mm_per_view[0]}, {self.mm_per_view[1]}, {self.mm_per_view[2]}] mm')
        print(f'Voxel Dims: [{self.vxl_dims[0]}, {self.vxl_dims[1]}, {self.vxl_dims[2]}] mm')
        print('')
    
    def init_file_image(self):
        
        def init_dcm_file():
            self.dcm = dicom.dcmread(file_path)
            self.original_X = self.dcm.pixel_array
            self.vxl_dims = 3*[self.dcm.PixelSpacing[0]]
        
        def init_tif_file():
            self.original_X = imread(self.file_properties['path'])
            self.original_X = np.flip(self.original_X, axis=0)
            self.vxl_dims = 3*[self.file_properties['vxl_dim_size']]
        
        file_path = self.file_properties['path']
        
        if file_path.lower().endswith('.dcm'):
            init_dcm_file()
        
        elif file_path.lower().endswith('.tif'):
            init_tif_file()
            
        else:
            raise RuntimeError("Provided file image type not supported. Must be of type .dcm or .tif!")
            
    def init_dir_image(self):
        
        def init_dcm_dir():
            plots = []
            for idx, file_name in enumerate(files):
                self.dcm = dicom.dcmread(file_name)
                pix = self.dcm.pixel_array
                plots.append(pix)
                
            self.original_X = np.stack(plots, axis=self.file_properties['transverse_axis'])
            self.vxl_dims = 3*[self.dcm.PixelSpacing[0]]
            
        def init_tif_dir():
            plots = []
            for idx, file_name in enumerate(files):
                plots.append(imread(file_name))

            self.original_X = np.stack(plots, axis=self.file_properties['transverse_axis'])
            self.vxl_dims = 3*[self.file_properties['vxl_dim_size']]
            
        files = glob.glob(self.file_properties['path'] + '*')
        
        if len(files) > 0:
        
            if files[0].lower().endswith('.dcm'):
                init_dcm_dir()
            
            elif files[0].lower().endswith('.tif'):
                init_tif_dir()
            
            else:
                raise RuntimeError("Provided image file(s) does not exist!")
                print(f"{files[0]}")
            
        else:
            raise RuntimeError("Provided image file directory is empty!")
            print(f"{self.file_properties['path']}")
    
    def create_dcm_object(self):
        dcm = dicom.Dataset()
        
        # Set required metadata
        dcm.PatientName = "Anonymous-Plant"
        dcm.PatientID = "123456"
        dcm.Modality = "CT"
        dcm.StudyInstanceUID = dicom.uid.generate_uid()
        dcm.SeriesInstanceUID = dicom.uid.generate_uid()
        dcm.SOPInstanceUID = dicom.uid.generate_uid()
        dcm.StudyDate = datetime.date.today().strftime("%Y%m%d")
        dcm.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        dcm.Manufacturer = "Python Generated"
        dcm.is_little_endian = True
        dcm.is_implicit_VR = False
        
        # Set Image Type
        dcm.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        dcm.PixelData = self.X.tobytes()
        dcm.Rows, dcm.Columns = self.X.shape[1], self.X.shape[2]
        dcm.NumberOfFrames = self.X.shape[0]  # Number of slices
        dcm.BitsAllocated = 16
        dcm.BitsStored = 16
        dcm.HighBit = 15
        dcm.SamplesPerPixel = 1
        dcm.PhotometricInterpretation = "MONOCHROME2"
        
        # Set voxel metadata
        dcm.SliceThickness = str(self.vxl_dims[0])
        dcm.PixelSpacing = [str(self.vxl_dims[1]), str(self.vxl_dims[2])]
        dcm.RescaleIntercept = str(self.intercept)
        dcm.RescaleSlope = str(self.slope)
        
        self.dcm = dcm
    
    def update_dcm_object(self):
        self.dcm.PixelData = self.X.tobytes()
        self.dcm.NumberOfFrames = self.X.shape[0]
        self.dcm.Rows, self.dcm.Columns = self.X.shape[1], self.X.shape[2]
        self.dcm.SliceThickness = str(self.vxl_dims[0])
        self.dcm.PixelSpacing = [str(self.vxl_dims[1]), str(self.vxl_dims[2])]
    
    # --- CHARACTERISTICS --- #
    
    def refresh_characteristics(self):
        self.set_vxls_per_view()
        self.set_mm_per_view()
    
    def set_vxls_per_view(self):
        self.vxls_per_view = [int(self.X.shape[0]), int(self.X.shape[1]), int(self.X.shape[2])]
    
    def set_mm_per_view(self):
        self.mm_per_view = np.multiply(self.vxl_dims, self.vxls_per_view)
    
    def get_extent(self, x_view, y_view):
        half_x = self.mm_per_view[x_view]/2
        half_y = self.mm_per_view[y_view]/2
        return [-half_x, half_x, -half_y, half_y]
    
    def get_matrix(self):
        return self.X.copy()
    
    # --- SLICE CONTROLS --- #
    
    def get_slice_number_from_mm(self, view, mm):
        slice_percent = (mm + self.mm_per_view[view]/2) / self.mm_per_view[view]
        if slice_percent < 0:
            return 0
        elif slice_percent > 1:
            return self.vxls_per_view[view] - 1
        else:
            return int(np.round(slice_percent * self.vxls_per_view[view]))
                
    def get_mm_from_slice_number(self, view, slice_number):
        if slice_number < 0:
            return 0
        elif slice_number >= self.vxls_per_view[view]:
            return self.vxls_per_view[view]
        else:
            slice_percent = slice_number / self.vxls_per_view[view]
            return slice_percent * self.mm_per_view[view] - self.mm_per_view[view]/2
        
    def preload_slices(self):
        self.preloaded_slices_1 = [self.X[i, :, :] for i in range(self.X.shape[0])]
        self.preloaded_slices_2 = [self.X[:, i, :] for i in range(self.X.shape[1])]
        self.preloaded_slices_3 = [self.X[:, :, i] for i in range(self.X.shape[2])]
        
    def get_slice_from_view(self, view):
        slice_number = self.slice_index[view]
        if view == 0:
            return self.preloaded_slices_1[slice_number]
        elif view == 1:
            return self.preloaded_slices_2[slice_number]
        else: # view == 2:
            return self.preloaded_slices_3[slice_number]
    
    def get_slice_from_mm(self, view, slice_mm):
        slice_number = self.get_slice_number_from_mm(view, slice_mm)
        if view == 0:
            return self.preloaded_slices_1[slice_number]
        elif view == 1:
            return self.preloaded_slices_2[slice_number]
        else: # view == 2:
            return self.preloaded_slices_3[slice_number]
    
    def get_slice_from_slice_number(self, view, slice_number):
        if view == 0:
            return self.preloaded_slices_1[slice_number]
        elif view == 1:
            return self.preloaded_slices_2[slice_number]
        else: # view == 2:
            return self.preloaded_slices_3[slice_number]
    
    def set_slice_from_mm(self, view, slice_mm):
        self.slice_index[view] = self.get_slice_number_from_mm(view, slice_mm)
        self.slice_mm[view] = slice_mm
        
    def set_slice_from_slice_number(self, view, slice_number):
        self.slice_index[view] = slice_number
        self.slice_mm[view] = self.get_mm_from_slice_number(view, slice_number)
        
    # --- TRANSFORMS --- #
    
    def invert_view(self, view):
        self.X = np.flip(self.X, axis=view)
        self.preload_slices()
        
    def rotate_view(self, view, direction):
        
        def rotate_slice_indices(x_view, y_view, z_view):
            
            # current slice indices
            slice_0_index = self.slice_index[TRANSVERSE]
            slice_1_index = self.slice_index[CORONAL]
            slice_2_index = self.slice_index[SAGITTAL]
            
            x_slice_index = self.slice_index[x_view]
            y_slice_index = self.slice_index[y_view]
            x = self.get_mm_from_slice_number(x_slice_index, z_view)
            y = self.get_mm_from_slice_number(y_slice_index, z_view)

            if direction == 'clockwise':
                t = y
                y = -x
                x = t
            else: # direction == 'anticlockwise':
                t = y
                y = x
                x = -t

            if z_view == 0:
                # x -> y = sagittal (2)
                # y -> x = coronal (1)
                slice_1_index = self.get_slice_number_from_mm(y, SAGITTAL)
                slice_2_index = self.get_slice_number_from_mm(x, CORONAL)
                
            elif z_view == 1:
                # x -> y = sagittal (2)
                # y -> x = transverse (0)
                slice_0_index = self.get_slice_number_from_mm(y, SAGITTAL)
                slice_2_index = self.get_slice_number_from_mm(x, TRANSVERSE)
                
            else: # z_view == 2:
                # x -> y coronal (1)
                # y -> x transverse (0)
                slice_0_index = self.get_slice_number_from_mm(y. CORONAL)
                slice_1_index = self.get_slice_number_from_mm(x, TRANSVERSE)
            
            self.slice_index[0] = slice_0_index
            self.slice_index[1] = slice_1_index
            self.slice_index[2] = slice_2_index
        
        TRANSVERSE = 0
        CORONAL = 1
        SAGITTAL = 2
        
        k = 1 if direction == 'clockwise' else 3
        
        # rearrange slice numbers        
        if view == 0: # transeverse
            # x is sagittal (2), y is coronal (1)
            self.X = np.rot90(self.X, k=k, axes=(1, 2))
            rotate_slice_indices(SAGITTAL, CORONAL, TRANSVERSE)
        
        elif view == 1: # coronal
            # x is sagittal (2), y is transverse (0)
            self.X = np.rot90(self.X, k=k, axes=(0, 2))
            rotate_slice_indices(SAGITTAL, TRANSVERSE, CORONAL)
            
        else: # view == 2: # sagittal
            # x is coronal (1), y is transverse (0)
            self.X = np.rot90(self.X, k=k, axes=(0, 1))
            rotate_slice_indices(CORONAL, TRANSVERSE, SAGITTAL)

        self.set_vxls_per_view()
        self.set_mm_per_view()
        self.update_slices()