import numpy as np
import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderV
from matplotlib.pyplot import colormaps


class ImageControls:
    def __init__(self, parent_frame, app, panel_views):
        self.app = app
        self.panel_views = panel_views

        # view controls
        self.tab_view = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_view, text='View')
        
        slice_controls = tk.Frame(self.tab_view, bd=1, relief=tk.SUNKEN)
        slice_controls.pack(anchor='n', side='left', padx=5, pady=2)
        slice_sliders = tk.Frame(slice_controls)
        slice_sliders.pack(side='top')
        self.views_slice_index = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        self.slider_view_1 = self.make_slider(slice_sliders, view=0, name='V1')
        self.slider_view_2 = self.make_slider(slice_sliders, view=1, name='V2')
        self.slider_view_3 = self.make_slider(slice_sliders, view=2, name='V3')
        self.make_linked_images(slice_controls)
        
        intensity_controls = tk.Frame(self.tab_view, bd=1, relief=tk.SUNKEN)
        intensity_controls.pack(anchor='n', side='left', padx=5, pady=2)
        self.intensity_limits = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=1.0)]
        self.last_intensity_limits = [0, 1]
        self.intensity_slider = self.make_rangeslider(intensity_controls)
        
        color_options = [
            "gist_yarg",
            "gist_gray",
            "inferno",
            "viridis",
            "hot",
            "jet"
        ]
        
        self.color_scheme = tk.StringVar()
        self.color_scheme.set(color_options[0])
        self.color_scheme.trace_add('write', self.set_colormap)
        drop = tk.OptionMenu(intensity_controls , self.color_scheme, *color_options)
        drop.config(width=7, anchor='w')
        drop.pack(side='top')
        
        # other controls
        self.tab_transform = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_transform, text='Transform')
        
        self.tab_saveload = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_saveload, text='Save/Load')
        
        self.tab_properties = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_properties, text='Properties') 
    
    def make_linked_images(self, parent):
        def checkbutton_command():
            if self.linked_images.get() == 1:
                self.app.panel_1_controls.linked_images.set(1)
                self.app.panel_2_controls.linked_images.set(1)
                # self.app.panel_3_controls.linked_images.set(1)
                
                self.set_linked_view_slice(view=0, slice_number=self.views_slice_index[0].get()) 
                self.set_linked_view_slice(view=1, slice_number=self.views_slice_index[1].get()) 
                self.set_linked_view_slice(view=2, slice_number=self.views_slice_index[2].get())
            else:
                self.app.panel_1_controls.linked_images.set(0)
                self.app.panel_2_controls.linked_images.set(0)
                # self.app.panel_3_controls.linked_images.set(0)
        
        self.linked_images = tk.IntVar(value=1)
        self.linked_images_checkbutton = tk.Checkbutton(parent, text="Link Images", onvalue=1, offvalue=0,
                                                        variable=self.linked_images, command=checkbutton_command)
        self.linked_images_checkbutton.pack(anchor='w', side='bottom')
    
    def make_slider(self, parent_frame, view, name):
        def slider_command(slider_value):
            slider_value = int(slider_value)
            if self.linked_images.get() == 1:
                self.set_linked_view_slice(view, slider_value)
            else:
                self.set_view_slice(view, slider_value, 'by_number')
            
        slider_frame = tk.Frame(parent_frame)
        slider_frame.pack(side='left')
        
        upper_bound = self.panel_views[view].X.vxls_in_dim[view]-1
        slider = tk.Scale(slider_frame, variable=self.views_slice_index[view],
                          command=slider_command,
                          showvalue=False,
                          from_=0, to=upper_bound,
                          width=10, length=200, orient='vertical')
        slider_showvalue = tk.Label(slider_frame, textvariable=self.views_slice_index[view])
        slider_label = tk.Label(slider_frame, text=name, width=3)
        
        slider_showvalue.pack(side='top')
        slider.pack(side='top')
        slider_label.pack(side='top')
        
        slider.set(0)
        self.set_initial_view_slice(view, 0.0, 'by_percent')
        return slider
    
    def make_rangeslider(self, parent):
        slider_handle = tk.PhotoImage(file='images/slider_handle_10px.png')
        intensity_range_slider = RangeSliderV(parent, self.intensity_limits,
                                              show_value=False,auto=False,
                                              imageU=slider_handle, image_anchorU='s',
                                              imageL=slider_handle, image_anchorL='n',
                                              padY=3, Height=206, Width=50, line_width=10,
                                              min_val=0, max_val=1, step_size=0.03,
                                              line_s_color='gray50', line_color='#c8c8c8',
                                              bgColor='#f0f0f0')
        intensity_range_slider.forceValues([0, 1])
        self.intensity_limits[0].trace_add('write', self.set_intensity)
        self.intensity_limits[1].trace_add('write', self.set_intensity)

        self.intensity_UB_label = tk.Label(parent, text='1.00')
        self.intensity_LB_label = tk.Label(parent, text='0.00')
        self.intensity_UB_label.pack(side='top')
        intensity_range_slider.pack(side='top')
        self.intensity_LB_label.pack(side='top')
        
        return intensity_range_slider
    
    def set_view_slice(self, view, slice_indicator, mode, from_mc=False):
        # if mode == 'by_number':
        slice_number = slice_indicator
        if mode == 'by_percent':
            slice_number = int(np.max([0, np.round(slice_indicator * (self.panel_views[view].X.vxls_in_dim[view]-1))]))
        self.views_slice_index[view].set(slice_number)
        self.panel_views[view].set_slice(slice_indicator, mode)
        
        if not from_mc:
            v_mm = slice_number * self.panel_views[view].X.vxl_dims[view]
            self.app.multi_cursor.set_crosshair(view, v_mm)
      
    def set_initial_view_slice(self, view, slice_indicator, mode):
        # if mode == 'by_number':
        slice_number = slice_indicator
        if mode == 'by_percent':
            slice_number = int(np.max([0, np.round(slice_indicator * (self.panel_views[view].X.vxls_in_dim[view]-1))]))
        self.views_slice_index[view].set(slice_number)
        self.panel_views[view].set_slice(slice_indicator, mode)
        
    def set_linked_view_slice(self, view, slice_number):
        slice_percent = slice_number / (self.panel_views[view].X.vxls_in_dim[view]-1)
        
        self.app.panel_1_controls.set_view_slice(view, slice_percent, 'by_percent')
        self.app.panel_2_controls.set_view_slice(view, slice_percent, 'by_percent')
        
        self.update_dual_view()
      
    def set_intensity(self, ar_name, index, mode):            
        self.intensity_UB_label['text'] = '{0:.2f}'.format(self.intensity_limits[1].get())
        self.intensity_LB_label['text'] = '{0:.2f}'.format(self.intensity_limits[0].get())
        
        for panel_view in self.panel_views:
            panel_view.set_intensity([self.intensity_limits[0].get(), self.intensity_limits[1].get()])
            
        self.last_intensity_limits[0] = self.intensity_limits[0].get()
        self.last_intensity_limits[1] = self.intensity_limits[1].get()
        
        self.update_dual_view()
    
    def set_colormap(self, ar_name, index, mode):
        for panel_view in self.panel_views:
            panel_view.set_cmap(self.color_scheme.get())
            
        self.update_dual_view()
    
    def update_dual_view(self):
        for panel_view in self.app.image_3_views:
            panel_view.update_data()
            