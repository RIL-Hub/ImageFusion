import tkinter as tk
import pydicom as dicom
import numpy as np
from pydicom.uid import generate_uid

class ExportControls:
    def __init__(self, app, image_controls, parent_frame, image_data):
        self.app = app
        self.image_controls = image_controls
        self.image_data = image_data
        
        # --- Export Images --- #
        
        export_images_frame = tk.Frame(parent_frame, bd=1, relief=tk.SUNKEN)
        export_images_frame.pack(side='top', anchor='nw', padx=5, pady=5)
        
        options = [
            "Transverse",
            "Sagittal",
            "Coronal"
        ]
        self.export_view = tk.StringVar()
        self.make_export_dropdown("View", export_images_frame, options, self.export_view)
        
        options = [
            "100",
            "300",
            "400",
            "600",
            "800",
            "1200"
        ]
        self.export_dpi = tk.StringVar()
        self.make_export_dropdown("DPI", export_images_frame, options, self.export_dpi)
        
        options = [
            "png",
            "jpg",
            "svg",
            "eps"
        ]
        self.export_filetype = tk.StringVar()
        self.make_export_dropdown("Filetype", export_images_frame, options, self.export_filetype)
        
        self.make_button(export_images_frame, "Export Image", self.export_image)
        
        # --- Export DICOM --- #
        
        export_DICOM_frame = tk.Frame(parent_frame, bd=1, relief=tk.SUNKEN)
        export_DICOM_frame.pack(side='top', anchor='nw', padx=5, pady=5)
        
        self.make_button(export_DICOM_frame, "Export DICOM", self.export_DICOM)
        
    def make_export_dropdown(self, name, parent_frame, options, var):
        drop_frame = tk.Frame(parent_frame)
        drop_frame.pack(side='top', anchor='w')
        
        drop_label = tk.Label(drop_frame, text=name, width=7)
        drop_label.pack(side='left')
        var.set(options[0])
        drop_menu = tk.OptionMenu(drop_frame , var, *options)
        drop_menu.config(width=10, anchor='w')
        drop_menu.pack(side='left')
    
    def make_button(self, parent, label, command):
        button = tk.Button(
            parent, 
            text=label, 
            command=command,
            # activebackground="blue", 
            # activeforeground="white",
            anchor="center",
            bd=1,
            # bg="lightgray",
            cursor="hand2",
            # disabledforeground="gray",
            fg="black",
            font=("Arial", 8),
            # height=1,
            highlightbackground="black",
            highlightcolor="green",
            highlightthickness=2,
            justify="center",
            overrelief="raised",
            # padx=2,
            # pady=5,
            # width=15,
            wraplength=100)
        button.pack(anchor='nw', side='top', padx=5, pady=2)
        return button
    
    def export_image(self):
        filename=self.export_view.get()
        dpi = int(self.export_dpi.get())
        extension = "." + self.export_filetype.get()
        
        filetype_dict = {
            ".png": "PNG files",
            ".jpg": "JPEG files",
            ".svg": "SVG files",
            ".eps": "EPS files",
            ".*": "All files"
        }
        
        # Build the filetypes list dynamically
        filetypes = [(filetype_dict[extension], f"*{extension}")]
        filetypes += [
            (desc, f"*{ext}") for ext, desc in filetype_dict.items() if ext != extension
        ]
        
        # Open file dialog for save location
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension = extension,
            filetypes = filetypes,
            initialfile = filename
        )
        
        if file_path:  # If the user didn't cancel
            view_dict = {"Transverse": 0, "Sagittal": 1, "Coronal": 2}
            self.image_controls.panel_views[view_dict[self.export_view.get()]].fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
            
    def export_DICOM(self):
        
        self.image_data.update_dcm_object()
        export_dcm = self.image_data.dcm
        
        # Center Image
        [z, x, y] = self.image_data.mm_per_view
        export_dcm.ImagePositionPatient = [-x/2, -y/2, -z/2]
        
        # Scale Pixel Values
        # export_dcm.RescaleIntercept = '0.0'
        # export_dcm.RescaleSlope = '1.0'
        
        # Set Color Window to Full Scale
        min_val = np.min(self.image_data.X)
        max_val = np.max(self.image_data.X)
        export_dcm.WindowCenter = [str((min_val + max_val) / 2)]
        export_dcm.WindowWidth = [str(max_val - min_val)]
        export_dcm.VOILUTFunction = "LINEAR"
        
        export_dcm.PixelData = self.image_data.X.tobytes()
        
        export_dcm.BodyPartExamined = "OTHER"
        export_dcm.Modality = "OT"
        export_dcm.RescaleType  = "US"

        # Update UIDs to ensure a new unique dataset
        export_dcm.SOPInstanceUID = generate_uid()
        export_dcm.SeriesInstanceUID = generate_uid()
        export_dcm.StudyInstanceUID = generate_uid()    
        
        # Open file dialog for save location
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension = '.dcm',
            filetypes = [("DICOM Images", '*.dcm')],
            initialfile = "expoter_image_fusion"
        )
        
        if file_path:  # Ensure user didn't cancel the save dialog
            export_dcm.save_as(file_path)
            print(f"DICOM file saved at: {file_path}")