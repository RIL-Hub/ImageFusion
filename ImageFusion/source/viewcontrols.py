import numpy as np
import tkinter as tk
from RangeSlider.RangeSlider import RangeSliderV

class ViewControls:
    def __init__(self, app, image_controls, parent_frame):
        self.app = app
        self.image_controls = image_controls
        self.parent_frame = parent_frame
        
        top_frame = tk.Frame(self.parent_frame)
        top_frame.pack(side=tk.TOP, anchor='nw')
        
        bottom_frame = tk.Frame(self.parent_frame)
        bottom_frame.pack(side=tk.TOP, anchor='nw')
        
        # --- Slice Frame --- #
        
        slice_controls = tk.Frame(top_frame, bd=1, relief=tk.SUNKEN)
        slice_controls.pack(side='left', padx=5, pady=2)
        
        slice_sliders = tk.Frame(slice_controls)
        slice_sliders.pack(side='top')
        
        slice_title = tk.Label(slice_sliders, text="Slice Controls")
        slice_title.pack(side='top', anchor='n')
        self.views_slice_index = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        self.make_slider(slice_sliders, view=0, name='T')
        self.make_slider(slice_sliders, view=1, name='C')
        self.make_slider(slice_sliders, view=2, name='S')
        self.make_linked_images(slice_controls)
        
        # --- Intensity Frame --- #
        
        intensity_controls = tk.Frame(top_frame, bd=1, relief=tk.SUNKEN)
        intensity_controls.pack(side='left', padx=5, pady=2)
        
        intensity_title = tk.Label(intensity_controls, text="Intensity")
        intensity_title.pack(side='top', anchor='n')
        
        self.make_rangeslider(intensity_controls)
        self.make_image_cmap_dropdown(intensity_controls)
        
        # --- Cursor Frmae --- #
        
        cursor_controls = tk.Frame(bottom_frame, bd=1, relief=tk.SUNKEN)
        cursor_controls.pack(side='left', padx=5, pady=2)
        
        cursor_title = tk.Label(cursor_controls, text="Cursor Controls")
        cursor_title.pack(side='top', anchor='n')
        
        cursor_top_frame = tk.Frame(cursor_controls)
        cursor_top_frame.pack(side=tk.TOP, anchor='nw')
        
        cursor_bottom_frame = tk.Frame(cursor_controls)
        cursor_bottom_frame.pack(side=tk.TOP, anchor='nw')
        
        self.make_cursor_toggle(cursor_top_frame)
        self.make_cursor_color_dropdown(cursor_top_frame)
        self.make_cursor_alpha_slider(cursor_bottom_frame)
        
    def make_slider(self, parent_frame, view, name):
        def slider_command(slider_value):
            slider_value = int(slider_value)
            self.set_view_slice(view, slider_value, 'by_number')
        
        def on_mouse_wheel(event):
            v = slider.get() - np.sign(event.delta)
            self.set_view_slice(view, v, 'by_number')
                
        slider_frame = tk.Frame(parent_frame)
        slider_frame.pack(side='left')
        
        upper_bound = self.image_controls.panel_views[view].X.vxls_in_dim[view]-1
        slider = tk.Scale(slider_frame, variable=self.views_slice_index[view],
                          command=slider_command,
                          showvalue=False,
                          from_=0, to=upper_bound,
                          width=10, length=200, orient='vertical')
        slider.bind("<MouseWheel>", on_mouse_wheel)
        slider_showvalue = tk.Label(slider_frame, textvariable=self.views_slice_index[view])
        slider_label = tk.Label(slider_frame, text=name, width=3)
        
        slider_label.pack(side='top')
        slider.pack(side='top')
        slider_showvalue.pack(side='top')
        
        slider.set(0)
        return slider
    
    def make_linked_images(self, parent):
        def checkbutton_command():
            if self.linked_images.get() == 1:
                self.app.panel_1_controls.view_controls.linked_images.set(1)
                self.app.panel_2_controls.view_controls.linked_images.set(1)
                
                self.set_view_slice(view=0, slice_indicator=self.views_slice_index[0].get(), mode='by_number') 
                self.set_view_slice(view=1, slice_indicator=self.views_slice_index[1].get(), mode='by_number') 
                self.set_view_slice(view=2, slice_indicator=self.views_slice_index[2].get(), mode='by_number')
            else:
                self.app.panel_1_controls.view_controls.linked_images.set(0)
                self.app.panel_2_controls.view_controls.linked_images.set(0)
        
        self.linked_images = tk.IntVar(value=1)
        self.linked_images_checkbutton = tk.Checkbutton(parent, text="Link Images", onvalue=1, offvalue=0,
                                                        variable=self.linked_images, command=checkbutton_command)
        self.linked_images_checkbutton.pack(anchor='w', side=tk.LEFT)
    
    def make_rangeslider(self, parent):
        self.intensity_limits = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=1.0)]
        self.last_intensity_limits = [0, 1]
        
        slider_handle = tk.PhotoImage(file='images/slider_handle_10px.png')
        intensity_range_slider = RangeSliderV(parent, self.intensity_limits,
                                              show_value=False,auto=False,
                                              imageU=slider_handle, image_anchorU='s',
                                              imageL=slider_handle, image_anchorL='n',
                                              padY=3, Height=200, Width=50, line_width=10,
                                              min_val=0, max_val=1, step_size=0.01,
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
    
    def make_image_cmap_dropdown(self, parent):
        cmap_options = [
            "gist_yarg",
            "gist_gray",
            "inferno",
            "viridis",
            "hot",
            "jet"
        ]
        self.color_scheme = tk.StringVar()
        self.color_scheme.set(cmap_options[0])
        self.color_scheme.trace_add('write', self.set_colormap)
        intensity_drop = tk.OptionMenu(parent , self.color_scheme, *cmap_options)
        intensity_drop.config(width=7, anchor='w')
        intensity_drop.pack(side='top')
    
    def make_cursor_toggle(self, parent):
        def checkbutton_command():
            if self.cursor_toggle.get() == 1:
                for panel_view in self.image_controls.panel_views:
                    panel_view.cursor_h.set_visible(True)
                    panel_view.cursor_v.set_visible(True)
            else:
                for panel_view in self.image_controls.panel_views:
                    panel_view.cursor_h.set_visible(False)
                    panel_view.cursor_v.set_visible(False)
            
            self.app.refresh_graphics()
        
        self.cursor_toggle = tk.IntVar(value=1)
        self.cursor_checkbutton = tk.Checkbutton(parent, text="", onvalue=1, offvalue=0, 
                                                        variable=self.cursor_toggle, command=checkbutton_command,
                                                        padx=0, pady=0)
        self.cursor_checkbutton.pack(side=tk.LEFT, anchor='w')
    
    def make_cursor_color_dropdown(self, parent):
        def update_cursor_color(event):
            color = cursor_color_options[self.cursor_color.get()]
            
            for panel_view in self.image_controls.panel_views:
                panel_view.cursor_h.set_color(color)
                panel_view.cursor_v.set_color(color)
            self.app.refresh_graphics()
        
        cursor_color_options = {
            'black': (0.0, 0.0, 0.0),
            'white': (1.0, 1.0, 1.0),
            'gray':  (0.5, 0.5, 0.5),
            'red':   (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue':  (0.0, 0.0, 1.0)
            }
        self.cursor_color = tk.StringVar()
        self.cursor_color.set(list(cursor_color_options.keys())[0])
        cursor_drop = tk.OptionMenu(parent , self.cursor_color, *cursor_color_options.keys(), command = update_cursor_color)
        cursor_drop.config(width=7, anchor='w')
        cursor_drop.pack(side=tk.LEFT, anchor='n', pady=0, padx=0)
     
    def make_cursor_alpha_slider(self, parent):
        def update_cursor_alpha(event):
            for panel_view in self.image_controls.panel_views:
                panel_view.cursor_h.set_alpha(slider.get())
                panel_view.cursor_v.set_alpha(slider.get())
            self.app.refresh_graphics()
        
        def on_mouse_wheel(event):
            alpha = slider.get() + np.sign(event.delta) * slider_step
            slider.set(alpha)
        
        slider_step = 0.1
        slider = tk.Scale(parent,
                          command=update_cursor_alpha,
                          showvalue=False,
                          from_=0.0, to=1.0, resolution=slider_step, sliderlength=15,
                          width=10, length=84, orient='horizontal')
        slider.bind("<MouseWheel>", on_mouse_wheel)
        slider.set(0.5)
        slider_label = tk.Label(parent, text='α', width=3, padx=0, pady=0)
        
        slider_label.pack(side=tk.LEFT, anchor='w')
        slider.pack(side=tk.LEFT, anchor='w')
        
    def set_intensity(self, ar_name, index, mode):            
        self.intensity_UB_label['text'] = '{0:.2f}'.format(self.intensity_limits[1].get())
        self.intensity_LB_label['text'] = '{0:.2f}'.format(self.intensity_limits[0].get())
        
        for panel_view in self.image_controls.panel_views:
            panel_view.set_intensity([self.intensity_limits[0].get(), self.intensity_limits[1].get()])
            
        self.last_intensity_limits[0] = self.intensity_limits[0].get()
        self.last_intensity_limits[1] = self.intensity_limits[1].get()
        
        self.update_dual_view()
    
    def set_colormap(self, ar_name, index, mode):
        for panel_view in self.image_controls.panel_views:
            panel_view.set_cmap(self.color_scheme.get())
            
        self.update_dual_view()
        
    def set_view_slice(self, view, slice_indicator, mode, from_self=False):
        
        if mode == 'by_number':
            slice_number = slice_indicator
        elif mode == 'by_percent':
            slice_number = int(np.max([0, np.round(slice_indicator * (self.image_controls.panel_views[view].X.vxls_in_dim[view]-1))]))
        
        self.views_slice_index[view].set(slice_number)
        self.image_controls.panel_views[view].set_slice(slice_indicator, mode)
        self.app.multi_cursor.update_crosshairs()
            
        if self.linked_images.get() == 1 and not from_self:
            slice_percent = slice_number / (self.image_controls.panel_views[view].X.vxls_in_dim[view]-1)
            
            self.app.panel_1_controls.view_controls.set_view_slice(view, slice_percent, 'by_percent', from_self=True)
            self.app.panel_2_controls.view_controls.set_view_slice(view, slice_percent, 'by_percent', from_self=True)
            
        self.update_dual_view()
        
    def update_dual_view(self):
        for panel_view in self.app.image_3_views:
            panel_view.update_data()