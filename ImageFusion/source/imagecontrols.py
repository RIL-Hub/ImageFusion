import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import affine_transform
from source.viewcontrols import ViewControls
from source.transformcontrols import TransformControls

class ImageControls:
    def __init__(self, parent_frame, app, panel_views):
        self.app = app
        self.panel_views = panel_views
        self.parent_frame = parent_frame

        # view controls
        self.tab_view = ttk.Frame(self.parent_frame)
        self.parent_frame.add(self.tab_view, text='View')
        self.view_controls = ViewControls(self.app, self, self.tab_view)
        
        # transform controls
        self.tab_transform = ttk.Frame(self.parent_frame)
        self.parent_frame.add(self.tab_transform, text='Transform')
        self.transform_controls = TransformControls(self.app, self, self.tab_transform)
        
        # other controls
        self.tab_saveload = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_saveload, text='Save/Load')
        
        self.tab_properties = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_properties, text='Properties') 