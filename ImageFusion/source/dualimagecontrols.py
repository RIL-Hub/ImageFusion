import numpy as np
import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderV
from matplotlib.pyplot import colormaps


class DualImageControls:
    def __init__(self, parent_frame, app, image_views):
        self.app = app
        self.image_views = image_views
        
        # view controls
        self.tab_view = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_view, text='View')
        
        opacity_controls = tk.Frame(self.tab_view, bd=1, relief=tk.SUNKEN)
        opacity_controls.pack(anchor='n', side='left', padx=5, pady=2)
        opacity_slider_frame = tk.Frame(opacity_controls)
        opacity_slider_frame.pack(side='top')
        self.opacity = tk.DoubleVar(value=0.5)
        self.opacity_slider = self.make_slider(opacity_slider_frame, name='α')
        
        # other controls
        self.tab_transform = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_transform, text='Transform')
        
        self.tab_saveload = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_saveload, text='Save/Load')
        
        self.tab_properties = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_properties, text='Properties') 
    
    
    def make_slider(self, parent_frame, name):
        def slider_command(slider_value):
            slider_showvalue['text'] = '{0:.2f}'.format(self.opacity.get())
            self.update_dual_view()

        slider_frame = tk.Frame(parent_frame)
        slider_frame.pack(side='left')
        
        slider = tk.Scale(slider_frame, variable=self.opacity,
                          command=slider_command,
                          showvalue=False,
                          from_=0, to=1, resolution=0.05,
                          width=10, length=200, orient='vertical')
        slider_showvalue = tk.Label(slider_frame, text='0.50')
        slider_label = tk.Label(slider_frame, text=name, width=3)
        
        slider_showvalue.pack(side='top')
        slider.pack(side='top')
        slider_label.pack(side='top')
        
        return slider

    def update_dual_view(self):
        for image_view in self.app.image_3_views:
            image_view.opacity = float(self.opacity.get())
            image_view.update_data()
            image_view.draw()
