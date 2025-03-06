import tkinter as tk

# custom modules
from source.imagecontrols import ImageControls
from source.dualimagecontrols import DualImageControls
from source.scannerpanel import ScannerPanel
from source.imageview import ImageView
from source.imagedata import ImageData
from source.dualimageview import DualImageView

# TODO:
    # fix display issue for slope/inetrcept values
    # image properties
    # default image_data and file loading
    # export attenuation image

class App(tk.Tk):
    
    def __init__(self, title):
        
        # main setup
        super().__init__()
        self.title(title)
        self.geometry("1800x1200")
        self.minsize(800, 400)
        
        # --- Debug init --- #
        
        data_path = 'D:/Code/Code_Research/Projects/ImageFusion/data/'
        
        # PET Images (stored as 3D .tifs)
        
        # PET_path = 'PET_PLANT--TRIT71120_Grass_Plant_12202024_PET--RS.2025.01.16.12.05_Plant_Recon/'
        # PET_image_path = data_path + PET_path + 'after_Flush.E20..nCOT.RS.I50.F18.RS.tif'
        
        PET_path = 'PET_CAPS--TRIT71120_Grass_Plant_12202024_PET--cap_tubes_recon/'
        PET_image_path = data_path + PET_path + 'cap_tubes.E20..nCOT.RS.I50.F18.RS.tif'
        
        # PET_path = 'PET_PLANT--TRIT71120_Grass_Plant_12202024_PET--RSM.2025.02.04.16.03/'
        # PET_image_path = data_path + PET_path + 'after_Flush.E20..nCOT.RS..C11.I25.T2.F30.RS.M.tif'
        
        # PET_image_path = data_path + 'ALIGNED_CT--TRIT71120LR_Zeego_CBCT_Recon.dcm'
        
        file_properties = {"path": PET_image_path, "vxl_dim_size": 0.5}
        self.X_PET = ImageData(file_properties)
        
        # CT Images (stored as series of .IMA files)
        
        CT_dir_path = 'CT--TRIT71120LR_Zeego_CBCT_Recon/'
        CT_images_path = data_path + CT_dir_path
        
        file_properties = {"path": CT_images_path, "transverse_axis": 0}
        self.X_CT = ImageData(file_properties)

        # --- Initialize App --- #

        # Top menu
        self.config(menu=MenuBar(self))
        
        # Scanner panels
        self.panel_1 = ScannerPanel(self)
        self.panel_2 = ScannerPanel(self)
        self.panel_3 = ScannerPanel(self)
        
        # Image views
        self.image_1_view_1 = ImageView(self, self.panel_1.image_view_1, self.X_CT, view=0)
        self.image_1_view_2 = ImageView(self, self.panel_1.image_view_2, self.X_CT, view=1)
        self.image_1_view_3 = ImageView(self, self.panel_1.image_view_3, self.X_CT, view=2)
        self.image_1_views = [self.image_1_view_1, self.image_1_view_2, self.image_1_view_3]
        
        self.image_2_view_1 = ImageView(self, self.panel_2.image_view_1, self.X_PET, view=0)
        self.image_2_view_2 = ImageView(self, self.panel_2.image_view_2, self.X_PET, view=1)
        self.image_2_view_3 = ImageView(self, self.panel_2.image_view_3, self.X_PET, view=2)
        self.image_2_views = [self.image_2_view_1, self.image_2_view_2, self.image_2_view_3]
        
        self.image_3_view_1 = DualImageView(self, self.panel_3.image_view_1, self.image_1_view_1, self.image_2_view_1)
        self.image_3_view_2 = DualImageView(self, self.panel_3.image_view_2, self.image_1_view_2, self.image_2_view_2)
        self.image_3_view_3 = DualImageView(self, self.panel_3.image_view_3, self.image_1_view_3, self.image_2_view_3)
        self.image_3_views = [self.image_3_view_1, self.image_3_view_2, self.image_3_view_3]
        self.images_by_views = [self.image_1_views, self.image_2_views]
        
        # Panel controls
        self.panel_1_controls = ImageControls(self.panel_1.image_controls, self, self.image_1_views, self.X_CT)
        self.panel_2_controls = ImageControls(self.panel_2.image_controls, self, self.image_2_views, self.X_PET)
        self.panel_3_controls = DualImageControls(self.panel_3.image_controls, self, self.image_3_views)
        self.controls = [self.panel_1_controls, self.panel_2_controls]

        # --- Initialize Views --- #
        
        self.panel_1_controls.view_controls.set_view_slice(0, 0, 'by_mm', original_call=False)
        self.panel_1_controls.view_controls.set_view_slice(1, 0, 'by_mm', original_call=False)
        self.panel_1_controls.view_controls.set_view_slice(2, 0, 'by_mm', original_call=False)
        
        self.panel_2_controls.view_controls.set_view_slice(0, 0, 'by_mm', original_call=False)
        self.panel_2_controls.view_controls.set_view_slice(1, 0, 'by_mm', original_call=False)
        self.panel_2_controls.view_controls.set_view_slice(2, 0, 'by_mm', original_call=False)
        
        self.refresh_data()
        self.refresh_graphics()
        
        # run
        self.mainloop()
    
    def refresh_graphics(self):
        
        self.panel_1_controls.view_controls.update_crosshairs()
        self.panel_2_controls.view_controls.update_crosshairs()
        
        self.image_1_view_1.draw()
        self.image_1_view_2.draw()
        self.image_1_view_3.draw()
        
        self.image_2_view_1.draw()
        self.image_2_view_2.draw()
        self.image_2_view_3.draw()
        
        self.image_3_view_1.draw()
        self.image_3_view_2.draw()
        self.image_3_view_3.draw()
        
    def refresh_data(self):
        
        self.image_1_view_1.update_data()
        self.image_1_view_2.update_data()
        self.image_1_view_3.update_data()
        
        self.image_2_view_1.update_data()
        self.image_2_view_2.update_data()
        self.image_2_view_3.update_data()
        
        self.image_3_view_1.update_data()
        self.image_3_view_2.update_data()
        self.image_3_view_3.update_data()
        
    def reload_slices(self):
        
        self.image_1_view_1.reload_slice()
        self.image_1_view_2.reload_slice()
        self.image_1_view_3.reload_slice()
        
        self.image_2_view_1.reload_slice()
        self.image_2_view_2.reload_slice()
        self.image_2_view_3.reload_slice()

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
        
    def donothing(self) -> int:
        return 0

def main() -> int:
    App("ImageFusion4D")
    return 0

if __name__ == "__main__":
    main()