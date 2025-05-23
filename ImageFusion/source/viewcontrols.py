import numpy as np
import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderV
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ViewControls:
    
    def __init__(self, app, image_views, image_data, parent_frame):
    
        self.app = app
        self.image_data = image_data
        self.image_views = image_views
        self.parent_frame = parent_frame
        
        top_frame = tk.Frame(self.parent_frame)
        top_frame.pack(side=tk.TOP, anchor='nw')
        
        bottom_frame = tk.Frame(self.parent_frame)
        bottom_frame.pack(side=tk.TOP, anchor='nw')
        
        # --- Slice Frame --- #
        
        slice_controls = tk.Frame(top_frame, bd=1, relief=tk.SUNKEN)
        slice_controls.pack(side='left', anchor='nw', padx=5, pady=5)
        
        slice_sliders = tk.Frame(slice_controls)
        slice_sliders.pack(side='top')
        
        slice_title = tk.Label(slice_sliders, text="Slice Controls")
        slice_title.pack(side='top', anchor='n')
        self.views_slice_index = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        self.make_slice_slider(slice_sliders, view=0, name='T')
        self.make_slice_slider(slice_sliders, view=1, name='C')
        self.make_slice_slider(slice_sliders, view=2, name='S')
        self.make_linked_images(slice_controls)
        
        # --- Display Frame --- #
        
        display_controls = tk.Frame(top_frame, bd=1, relief=tk.SUNKEN)
        display_controls.pack(side='left', anchor='nw', padx=5, pady=5)
        
        display_title = tk.Label(display_controls, text="Display Controls")
        display_title.pack(side='top', anchor='n')
        
        self.make_intensity_controller(display_controls)
        self.make_image_cmap_dropdown(display_controls)
        self.make_interpolation_dropdown(display_controls)
        
        # --- Cursor Frame --- #
        
        cursor_controls = tk.Frame(bottom_frame, bd=1, relief=tk.SUNKEN)
        cursor_controls.pack(side='left', padx=5, pady=5)
        
        cursor_title = tk.Label(cursor_controls, text="Cursor Controls")
        cursor_title.pack(side='top', anchor='n')
        
        cursor_top_frame = tk.Frame(cursor_controls)
        cursor_top_frame.pack(side=tk.TOP, anchor='nw')
        
        cursor_bottom_frame = tk.Frame(cursor_controls)
        cursor_bottom_frame.pack(side=tk.TOP, anchor='nw')
        
        self.make_cursor_toggle(cursor_top_frame)
        self.make_cursor_color_dropdown(cursor_top_frame)
        self.make_cursor_alpha_slider(cursor_bottom_frame)
        
        # --- Image Mouse Controls --- #
        
        for view, image_view in enumerate(self.image_views):
            image_view.fig.canvas.mpl_connect('button_press_event', lambda event, view=view: self.on_click(event, view))
            image_view.fig.canvas.mpl_connect('scroll_event', lambda event, view=view: self.on_scroll(event, view))
        
    # --- Functions for making objects --- #
      
    def make_slice_slider(self, parent_frame, view, name):
        
        def slider_command(slider_value):
            slider_value = int(slider_value)
            self.set_view_slice(view, slider_value, 'by_number')
        
        def on_mouse_wheel(event):
            v = slider.get() - np.sign(event.delta)
            self.set_view_slice(view, v, 'by_number')
                
        slider_frame = tk.Frame(parent_frame)
        slider_frame.pack(side='left')
        
        upper_bound = self.image_data.vxls_per_view[view]-1
        slider = tk.Scale(slider_frame, variable=self.views_slice_index[view],
                          command=slider_command,
                          showvalue=False,
                          from_=upper_bound, to=0,
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
        
        self.linked_images = tk.IntVar(value=0)
        self.linked_images_checkbutton = tk.Checkbutton(parent, text="Link Images", onvalue=1, offvalue=0,
                                                        variable=self.linked_images, command=checkbutton_command)
        self.linked_images_checkbutton.pack(anchor='w', side=tk.LEFT)
    
    def make_intensity_controller(self, parent):

        # --- wrapper frame --- #

        wrapper_frame = tk.Frame(parent)
        wrapper_frame.pack(side='top', anchor='w')
        
        # -- (left) verticle label -- #
        
        label_canvas = tk.Canvas(wrapper_frame, width=20, height=60)
        label_canvas.pack(side='left', anchor='w', padx=0, pady=0)
        label_canvas.create_text(10, 30, text="Intesnity", angle=90)
        
        # -- (middle) range slider -- #
        
        slider_frame = tk.Frame(wrapper_frame)
        slider_frame.pack(side='left', anchor='w', padx=0, pady=0)
        
        self.intensity_limits = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=1.0)]
        
        slider_handle = tk.PhotoImage(file='images/slider_handle_10px.png')
        intensity_range_slider = RangeSliderV(slider_frame, self.intensity_limits,
                                              show_value=False,auto=False,
                                              imageU=slider_handle, image_anchorU='s',
                                              imageL=slider_handle, image_anchorL='n',
                                              padY=3, font_size=-10, # negative font to trick min width requirements
                                              Height=200, Width=10, line_width=10,
                                              min_val=0, max_val=1, step_size=0.01,
                                              line_s_color='gray50', line_color='#c8c8c8',
                                              bgColor='#c8c8c8')
        
        intensity_range_slider.forceValues([0, 1])
        self.intensity_limits[0].trace_add('write', self.set_intensity)
        self.intensity_limits[1].trace_add('write', self.set_intensity)

        self.intensity_UB_label = tk.Label(slider_frame, text='1.00')
        self.intensity_LB_label = tk.Label(slider_frame, text='0.00')
        
        self.intensity_UB_label.pack(side='top')
        intensity_range_slider.pack(side='top')
        self.intensity_LB_label.pack(side='top')
        
        # -- (right) histogram -- #
        
        image = self.image_views[0].image_data.get_matrix().flatten()
        
        px = 1/plt.rcParams['figure.dpi'] # pixel in inches
        self.int_fig = plt.Figure(figsize=(50*px,200*px))
        self.int_ax = self.int_fig.add_subplot()

        self.int_ax.get_xaxis().set_ticks([])
        self.int_ax.get_yaxis().set_ticks([])
        self.int_ax.axis('off')
        self.int_ax.set_xscale('log')
        self.int_fig.tight_layout(pad=0)
        
        counts, bins, self.patches = self.int_ax.hist(image, bins=100, orientation='horizontal', color='black')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        centered_bins = bin_centers - min(bin_centers)
        self.scaled_bins = centered_bins/max(centered_bins)
        
        cm = plt.cm.get_cmap('gist_yarg')
        for sb, p in zip(self.scaled_bins, self.patches):
            plt.setp(p, 'facecolor', cm(sb))
    
        self.int_ax.set_ylim([0, bins[-1]])
        
        self.hist_canvas = FigureCanvasTkAgg(self.int_fig, master=wrapper_frame)
        self.hist_canvas.get_tk_widget().pack(side='left', padx=0, pady=0)
        
        return intensity_range_slider
     
    def make_image_cmap_dropdown(self, parent):
        cmap_options = [
            "gist_yarg",
            "gist_gray",
            "inferno",
            "viridis",
            "hot",
            "jet",
            "Reds"
        ]
        
        self.color_scheme = tk.StringVar()
        self.color_scheme.set(cmap_options[0])
        self.color_scheme.trace_add('write', self.set_colormap)
        
        _ = tk.StringVar()
        cmap_drop = ttk.OptionMenu(parent, _, "Color Map")
        cmap_drop.pack(side='top', anchor='w')
        cmap_drop.config(width=12)
        menu = cmap_drop["menu"]
        
        for cmap in cmap_options:
            menu.add_radiobutton(label=cmap, variable=self.color_scheme, value=cmap)      
    
    def make_interpolation_dropdown(self, parent):
        
        interpolation_options = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        
        self.interpolation = tk.StringVar()
        self.interpolation.set(interpolation_options[0])
        self.interpolation.trace_add('write', self.set_interpolation)

        _ = tk.StringVar()
        interpolation_drop = ttk.OptionMenu(parent, _, "Interpolation")
        interpolation_drop.pack(side='top', anchor='w')
        interpolation_drop.config(width=12)
        menu = interpolation_drop["menu"]
        
        for interpolation in interpolation_options:
            menu.add_radiobutton(label=interpolation, variable=self.interpolation, value=interpolation) 
    
    def make_cursor_toggle(self, parent):
        
        def checkbutton_command():
            
            if self.cursor_toggle.get() == 1:
                for image_view in self.image_views:
                    image_view.cursor_h.set_visible(True)
                    image_view.cursor_v.set_visible(True)
                    image_view.update_data()
                    
            else:
                for image_view in self.image_views:
                    image_view.cursor_h.set_visible(False)
                    image_view.cursor_v.set_visible(False)
                    image_view.update_data()
                    
            self.app.refresh_graphics()
        
        self.cursor_toggle = tk.IntVar(value=1)
        self.cursor_checkbutton = tk.Checkbutton(parent, text="", onvalue=1, offvalue=0, 
                                                        variable=self.cursor_toggle, command=checkbutton_command,
                                                        padx=0, pady=0)
        self.cursor_checkbutton.pack(side=tk.LEFT, anchor='w')
    
    def make_cursor_color_dropdown(self, parent):
        
        def update_cursor_color(event):
            
            color = cursor_color_options[self.cursor_color.get()]
            for image_view in self.image_views:
                image_view.cursor_h.set_color(color)
                image_view.cursor_v.set_color(color)
                image_view.update_data()
                
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
            
            for image_view in self.image_views:
                image_view.cursor_h.set_alpha(slider.get())
                image_view.cursor_v.set_alpha(slider.get())
            
                image_view.update_data()
            self.app.refresh_graphics()
        
        def on_mouse_wheel(event):
            alpha = slider.get() + np.sign(event.delta) * slider_step
            slider.set(alpha)
        
        slider_step = 0.01
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
    
    # --- Functions for setting view parameters --- #

    def set_intensity(self, ar_name, index, mode):            
        self.intensity_UB_label['text'] = '{0:.2f}'.format(self.intensity_limits[1].get())
        self.intensity_LB_label['text'] = '{0:.2f}'.format(self.intensity_limits[0].get())
        
        for image_view in self.image_views:
            image_view.set_intensity([self.intensity_limits[0].get(), self.intensity_limits[1].get()])
        
        self.set_intensity_colors()
        self.update_dual_view()
        self.app.refresh_graphics()
    
    def set_colormap(self, ar_name, index, mode):
        cmap = self.color_scheme.get()
        for image_view in self.image_views:
            image_view.set_cmap(cmap)
        self.set_intensity_colors()
        self.update_dual_view()
        self.app.refresh_graphics()
    
    def set_intensity_colors(self):
        cm = plt.cm.get_cmap(self.color_scheme.get())
        
        local_scaled_bins = np.array(self.scaled_bins.copy())
        local_scaled_bins[local_scaled_bins < self.intensity_limits[0].get()] = 0
        local_scaled_bins[local_scaled_bins > self.intensity_limits[1].get()] = 1
        
        lo_ind = np.where(local_scaled_bins==0)[0][-1]
        hi_ind = np.where(local_scaled_bins==1)[0][0]+1

        local_scaled_bins[lo_ind:hi_ind] = np.linspace(0, 1, num=hi_ind-lo_ind)
        
        for sb, p in zip(local_scaled_bins, self.patches):
            plt.setp(p, 'facecolor', cm(sb))
            
        self.int_fig.patch.set_facecolor(cm(0))
        self.hist_canvas.draw()
    
    def set_interpolation(self, ar_name, index, mode):
        for image_view in self.image_views:
            image_view.set_interpolation(self.interpolation.get())
            
        self.update_dual_view()
        self.app.refresh_graphics()
    
    # --- Slice controls --- #
    
    def set_view_slice(self, view, slice_indicator, mode, original_call=True):
        
        if mode == 'by_number':
            slice_number = slice_indicator
            slice_mm = self.image_data.get_mm_from_slice_number(view, slice_number)
        else: # mode == 'by_mm':
            slice_mm = slice_indicator
            slice_number = self.image_data.get_slice_number_from_mm(view, slice_mm)
            
        if self.linked_images.get() == 1 and original_call:
            self.app.panel_1_controls.view_controls.set_view_slice(view, slice_mm, 'by_mm', original_call=False)
            self.app.panel_2_controls.view_controls.set_view_slice(view, slice_mm, 'by_mm', original_call=False)
            original_call = True
        else:            
            self.image_views[view].slice = self.image_data.get_slice_from_mm(view, slice_mm)
            self.image_data.set_slice_from_mm(view, slice_mm)
            self.views_slice_index[view].set(slice_number)
        
        if original_call:
            self.update_dual_view()
            self.app.refresh_data()
            self.app.refresh_graphics()
    
    def on_click(self, event, view):
        
        if event.button == 1 and event.xdata and event.ydata:

            event_x = event.xdata
            event_y = event.ydata
            
            TRANVERSE = 0
            CORONAL   = 1
            SAGITTAL  = 2
            mode = 'by_mm'
            
            if view == TRANVERSE:
                # clicked on transverse image, it doesn't change
                # event_x is sagittal, event_y is coronal
                self.set_view_slice(SAGITTAL, event_x, mode)
                self.set_view_slice(CORONAL, event_y, mode)
            
            elif view == CORONAL:
                # clicked on coronal image, it doesn't change
                # event_x is sagittal, event_y is transverse
                self.set_view_slice(SAGITTAL, event_x, mode)
                self.set_view_slice(TRANVERSE, event_y, mode)
                
            else: # view == SAGITTAL:
                # clicked on sagittal image, it doesn't change
                # event_x is coronal, event_y is transverse
                self.set_view_slice(CORONAL, event_x, mode)
                self.set_view_slice(TRANVERSE, event_y, mode)
                
    def on_scroll(self, event, view):
        
        v = self.views_slice_index[view].get()
                
        if event.button == 'up':
            if v < self.image_views[0].image_data.vxls_per_view[view]-1:    
                v = v + 1
        else: # event.button == 'down':
            if v > 0:            
                v = v - 1
                
        self.set_view_slice(view, v, 'by_number')
            
    def update_crosshairs(self):
        
        TRANVERSE = 0
        CORONAL   = 1
        SAGITTAL  = 2
        
        for view in [0, 1, 2]:
        
            if view == 0: # transverse
                # x is sagittal, y is coronal
                x = self.image_data.slice_mm[SAGITTAL]
                y = self.image_data.slice_mm[CORONAL]
                
            elif view == 1: # coronal
                # x is sagittal, y is transverse
                x = self.image_data.slice_mm[SAGITTAL]
                y = self.image_data.slice_mm[TRANVERSE]
            
            else: # view == 2: # sagittal
                # x is coronal, y is transverse
                x = self.image_data.slice_mm[CORONAL]
                y = self.image_data.slice_mm[TRANVERSE]

            self.image_views[view].cursor_h.set_ydata(y)
            self.image_views[view].cursor_v.set_xdata(x)
    
    def update_dual_view(self):
        for image_view in self.app.image_3_views:
            image_view.update_data()