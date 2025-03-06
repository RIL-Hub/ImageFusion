import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import affine_transform
from source.viewcontrols import ViewControls
from source.transformcontrols import TransformControls
from source.exportcontrols import ExportControls

class ImageControls:
    def __init__(self, parent_frame, app, image_views, image_data):
        self.app = app
        self.image_views = image_views
        self.parent_frame = parent_frame
        self.image_data = image_data

        # view controls
        self.tab_view = ttk.Frame(self.parent_frame)
        self.parent_frame.add(self.tab_view, text='View')
        self.view_controls = ViewControls(self.app, self.image_views, self.image_data, self.tab_view)
        
        # transform controls
        self.tab_transform = ttk.Frame(self.parent_frame)
        self.parent_frame.add(self.tab_transform, text='Transform')
        self.transform_controls = TransformControls(self.app, self.image_data, self, self.tab_transform)
        
        # export controls
        self.tab_export = ttk.Frame(self.parent_frame)
        self.parent_frame.add(self.tab_export, text='Export')
        self.export_controls = ExportControls(self.app, self, self.tab_export, self.image_data)
        
        # properties controls
        self.tab_properties = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_properties, text='Properties') 