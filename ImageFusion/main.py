import numpy as np
import tkinter as tk

# custom modules
from source.imagecontrols import ImageControls
from source.dualimagecontrols import DualImageControls
from source.scannerpanel import ScannerPanel
from source.imageviews import ImageData, ImageView
from source.dualimageviews import DualImageData, DualImageView

# TODO:
# opacity slider for dual view
# multicursors

class App(tk.Tk):
    
    def __init__(self, title):
        # main setup
        super().__init__()
        self.title(title)
        self.geometry("1600x1000")
        self.minsize(800,400)
        
        # debug init
        data_path = 'D:/Code/Code_Research/Projects/CT2Attenuation/'

        # PET Images (stored as 3D .tifs)
        PET_path = 'nasir_recon-20240606T222236Z-003/'
        PET_image_path = data_path + PET_path + 'USCS_Orgain_Soil_4Cap_150uC_fud_12uCi_1hour_11_53am.E15.V0.5I50RAW.tif'
        file_properties = {"path": PET_image_path, "vxl_dim1_size": 0.5, "vxl_dim2_size": 0.5, "vxl_dim3_size": 0.5}
        self.X_PET = ImageData(file_properties, image_type='PET')
        
        # CT Images (stored as series of .IMA files)
        CT_dir_path = '2024-05-23_EARTH_SHOT_CAP_TUBES_BENNETT-20240606T205140Z-001/'
        CT_images_path = data_path + CT_dir_path + 'DYNACT_HEAD_NAT_FILL_HU_NORMAL_180UM_VOXEL_109KV/'
        file_properties = {"path": CT_images_path}
        self.X_CT = ImageData(file_properties, image_type='CT')
        
        self.X_dual = DualImageData(self.X_CT, self.X_PET)
        
        self.match_image_dims()
        self.X_PET.print_info()
        self.X_CT.print_info()
        
        # top menu
        self.config(menu=MenuBar(self))
        
        # scanner panels
        self.panel_1 = ScannerPanel(self)
        self.panel_2 = ScannerPanel(self)
        self.panel_3 = ScannerPanel(self)
        
        # image views
        self.image_1_view_1 = ImageView(self.panel_1.image_view_1, self.X_CT, view=0)
        self.image_1_view_2 = ImageView(self.panel_1.image_view_2, self.X_CT, view=1)
        self.image_1_view_3 = ImageView(self.panel_1.image_view_3, self.X_CT, view=2)
        self.image_1_views = [self.image_1_view_1, self.image_1_view_2, self.image_1_view_3]
        
        self.image_2_view_1 = ImageView(self.panel_2.image_view_1, self.X_PET, view=0)
        self.image_2_view_2 = ImageView(self.panel_2.image_view_2, self.X_PET, view=1)
        self.image_2_view_3 = ImageView(self.panel_2.image_view_3, self.X_PET, view=2)
        self.image_2_views = [self.image_2_view_1, self.image_2_view_2, self.image_2_view_3]
        
        self.image_3_view_1 = DualImageView(self.panel_3.image_view_1, self.image_1_view_1, self.image_2_view_1)
        self.image_3_view_2 = DualImageView(self.panel_3.image_view_2, self.image_1_view_2, self.image_2_view_2)
        self.image_3_view_3 = DualImageView(self.panel_3.image_view_3, self.image_1_view_3, self.image_2_view_3)
        self.image_3_views = [self.image_3_view_1, self.image_3_view_2, self.image_3_view_3]
        
        self.images_views = [self.image_1_views, self.image_2_views, self.image_3_views]
        
        # panel controls
        self.panel_1_controls = ImageControls(self.panel_1.image_controls, self, self.image_1_views)
        self.panel_2_controls = ImageControls(self.panel_2.image_controls, self, self.image_2_views)
        # self.panel_3_controls = DualImageControls(self.panel_3.image_controls, self, self.image_3_views)

        # run
        self.mainloop()
        
    def match_image_dims(self):
        self.max_dims = [np.max([dim1, dims2]) for (dim1, dims2) in zip(self.X_PET.dims, self.X_CT.dims)]
        self.X_PET.pad_to_dims(self.max_dims)
        self.X_CT.pad_to_dims(self.max_dims)
        
        self.vxls_per_dim = [-1, -1, -1]
        self.vxl_length_per_dim = [-1, -1, -1]
        for i in range(3):
            if self.X_PET.vxls_in_dim[i] > self.X_CT.vxls_in_dim[i]:
                self.vxls_per_dim[i] = self.X_PET.vxls_in_dim[i]
                self.vxl_length_per_dim[i] = self.X_PET.vxl_dims[i]
            else:
                self.vxls_per_dim[i] = self.X_CT.vxls_in_dim[i]
                self.vxl_length_per_dim[i] = self.X_CT.vxl_dims[i]
    
    def refresh_graphics(self):
        self.image_1_view_1.canvas.draw()
        self.image_1_view_2.canvas.draw()
        self.image_1_view_3.canvas.draw()
        
        self.image_2_view_1.canvas.draw()
        self.image_2_view_2.canvas.draw()
        self.image_2_view_3.canvas.draw()
        
        self.image_3_view_1.canvas.draw()
        self.image_3_view_2.canvas.draw()
        self.image_3_view_3.canvas.draw()

class MenuBar(tk.Menu):
    def __init__(self, parent):
        super().__init__(parent)
        
        file_menu = tk.Menu(self, tearoff=0)
        file_menu.add_command(label="New", command=self.donothing)
        file_menu.add_command(label="Open", command=self.donothing)
        file_menu.add_command(label="Save", command=self.donothing)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.donothing)
        self.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(self, tearoff=0)
        help_menu.add_command(label="Help Index", command=self.donothing)
        help_menu.add_command(label="About...", command=self.donothing)
        help_menu.add_command(label="Help", command=self.donothing)
        self.add_cascade(label="Help", menu=help_menu)
        
    def donothing(self):
        pass

def main() -> int:
    App("ImageFusion4D")
    return 0

if __name__ == "__main__":
    main()