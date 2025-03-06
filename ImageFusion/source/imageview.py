import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['figure.dpi'] = 100

class ImageView:
    
    def __init__(self, app, parent_frame, image_data, view):
        
        # Initialize Controls
        self.app = app
        self.view = view
        self.cmap = 'gist_yarg'
        self.interpolation = "None"
        self.enlarged_flag = False
        
        # Initialize Image
        self.image_data = image_data
        self.slice = self.image_data.get_slice_from_slice_number(view, 0)
        self.image_data.set_slice_from_slice_number(view, 0)
        self.intensity_limits = [0.0, self.image_data.max_intensity]

        # Create a new figure and axis and display the image
        self.fig = plt.Figure() # figsize=(3, 3)
        self.ax = self.fig.add_subplot()
        
        self.image = self.ax.imshow(self.slice,
                                    extent=self.get_extent(),
                                    vmin=0, vmax=self.intensity_limits[1],
                                    cmap=self.cmap,
                                    interpolation=self.interpolation,
                                    origin='lower')
        
        # Set title and axes
        titles = ['Transverse', 'Coronal', 'Sagittal']
        self.ax.set_title(titles[self.view])
        self.set_xaxis(self.ax)
        self.set_yaxis(self.ax)
        
        # Make colorbar
        divider_PET = make_axes_locatable(self.ax)
        cb_cax = divider_PET.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(self.image, cax=cb_cax)
        
        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')
        
        # Create cursor
        self.cursor_h = self.ax.axhline(y=[0], visible=True, color='black', alpha=0.5)
        self.cursor_v = self.ax.axvline(x=[0], visible=True, color='black', alpha=0.5)
        
        # Bind right-click event to open enlarged plot
        self.canvas.get_tk_widget().bind("<Button-2>", self.enlarge_plot)
        self.canvas.get_tk_widget().bind("<Button-3>", self.enlarge_plot)
        self.canvas.get_tk_widget().focus_set()  # Ensure it can capture key events  
    
    # --- View parameters --- #
    
    def get_extent(self):
        xy_views = [[2,1], [2, 0], [1, 0]]
        return self.image_data.get_extent(*xy_views[self.view])

    def set_intensity(self, intensity_limits):
        self.intensity_limits = [intensity_limits[0]*self.image_data.max_intensity, intensity_limits[1]*self.image_data.max_intensity]
        self.update_data()
    
    def set_cmap(self, cmap):
        self.cmap = cmap
        self.update_data()
    
    def set_interpolation(self, interpolation):
        self.interpolation = interpolation
        self.update_data()
    
    def set_xaxis(self, ax):
        def get_divisors(n):
            numbers = np.arange(1, n + 1) 
            divisors = numbers[n % numbers == 0] 
            count = len(divisors)
            return count, divisors
        
        x_dims = [2, 2, 1]
        x_dim = x_dims[self.view]
        
        dim_max = self.image_data.mm_per_view[x_dim]/2
        
        rounded_max = np.ceil(dim_max / 10) * 10
        count, divisors = get_divisors(rounded_max)
        
        if count == 1:
            tick_interval = divisors[0]
        elif count >= 3:
            tick_interval = divisors[count-3]
        else:
            tick_interval = divisors[count-2]
        
        positive_ticks = np.arange(0, rounded_max+1, tick_interval)
        negative_ticks = -positive_ticks[::-1]
        negative_ticks = negative_ticks[:-1] # remove redundant 0
        self.new_ticks_x = np.concatenate((negative_ticks, positive_ticks)).astype(int)
        
        self.original_positions_x = (self.new_ticks_x + dim_max) / (dim_max + dim_max) * self.image_data.vxls_per_view[x_dim]
        
        # ax.set_xticks(self.original_positions_x)
        # ax.set_xticklabels(self.new_ticks_x)
        
        x_labels = ['Sagittal (mm)', 'Sagittal (mm)', 'Coronal (mm)'] 
        x_label = x_labels[self.view]
        ax.set_xlabel(x_label)
    
    def set_yaxis(self, ax):
        
        def get_divisors(n):
            numbers = np.arange(1, n + 1) 
            divisors = numbers[n % numbers == 0] 
            count = len(divisors)
            return count, divisors
        
        y_dims = [1, 0, 0]
        y_dim = y_dims[self.view]
        
        dim_max = self.image_data.mm_per_view[y_dim]/2
        
        rounded_max = np.ceil(dim_max / 10) * 10
        count, divisors = get_divisors(rounded_max)
        
        if count == 1:
            tick_interval = divisors[0]
        elif count >= 3:
            tick_interval = divisors[count-3]
        else:
            tick_interval = divisors[count-2]
        
        positive_ticks = np.arange(0, rounded_max+1, tick_interval)
        negative_ticks = -positive_ticks[::-1]
        negative_ticks = negative_ticks[:-1] # remove redundant 0
        self.new_ticks_y = np.concatenate((negative_ticks, positive_ticks)).astype(int)
        
        self.original_positions_y = (self.new_ticks_y + dim_max) / (dim_max + dim_max) * self.image_data.vxls_per_view[y_dim]
        self.original_positions_y = self.original_positions_y[::-1]
        
        # ax.set_yticks(self.original_positions_y)
        # ax.set_yticklabels(self.new_ticks_y)
        
        y_labels = ['Coronal (mm)', 'Transverse (mm)', 'Transverse (mm)'] 
        y_label = y_labels[self.view]
        ax.set_ylabel(y_label)    
    
    # --- Drawing --- #
    
    def reload_slice(self):
        self.slice = self.image_data.get_slice_from_view(self.view)
    
    def update_data(self):        
        self.image.set_interpolation(self.interpolation)
        self.image.set_cmap(self.cmap)
        self.image.set_data(np.clip(self.slice, self.intensity_limits[0], self.intensity_limits[1]))
        self.image.set_clim(vmin=self.intensity_limits[0], vmax=self.intensity_limits[1])
        self.cbar.update_normal(self.image) 
        self.image.set_extent(self.get_extent())
        
        if self.enlarged_flag:
            self.update_enlarged_image()
    
    def draw(self):
        self.canvas.draw_idle()
        
        if self.enlarged_flag:
            self.enlarged_canvas.draw_idle()
    
    # --- Enlarged PLot --- #    
        
    def update_enlarged_image(self):
        
        self.enlarged_image.set_data(self.image.get_array())
        self.enlarged_image.set_interpolation(self.image.get_interpolation())
        self.enlarged_image.set_cmap(self.image.get_cmap())
        self.enlarged_image.set_clim(self.image.get_clim())
        self.enlarged_image.set_extent(self.image.get_extent())
        self.enlarged_cbar.update_normal(self.image)

        self.enlarged_cursor_h.set_ydata(self.cursor_h.get_ydata())
        self.enlarged_cursor_h.set_color(self.cursor_h.get_color())
        self.enlarged_cursor_h.set_alpha(self.cursor_h.get_alpha())
        self.enlarged_cursor_h.set_visible(self.cursor_h.get_visible())

        self.enlarged_cursor_v.set_xdata(self.cursor_v.get_xdata())
        self.enlarged_cursor_v.set_color(self.cursor_v.get_color())
        self.enlarged_cursor_v.set_alpha(self.cursor_v.get_alpha())
        self.enlarged_cursor_v.set_visible(self.cursor_v.get_visible())

    def enlarge_plot(self, event=None):
        
        def close_enlarged( popup):
            self.enlarged_flag = False
            popup.destroy()
        
        if not self.enlarged_flag:
            self.enlarged_flag = True
            
            popup = tk.Toplevel(self.app)
            popup.title("Zoomed Plot")
            popup.geometry("600x400")
            
            # Create a new figure and axis and display the image
            self.enlarged_fig = plt.Figure(dpi=100)
            self.enlarged_ax = self.enlarged_fig.add_subplot()
            self.enlarged_image = self.enlarged_ax.imshow(self.image.get_array(), extent=self.get_extent())
            
            # Set title and axes
            self.enlarged_ax.set_title(self.ax.get_title())
            self.enlarged_ax.set_xlabel(self.ax.get_xlabel())   
            self.enlarged_ax.set_ylabel(self.ax.get_ylabel())
            
            # Add colorbar
            divider_PET = make_axes_locatable(self.enlarged_ax)
            cb_cax = divider_PET.append_axes("right", size="5%", pad=0.05)
            self.enlarged_cbar = self.fig.colorbar(self.enlarged_image, cax=cb_cax)
            
            # Adjust layout
            self.enlarged_fig.tight_layout()
            popup.transient(self.app) # set to be on top of the main window
                        
            # Create the canvas
            self.enlarged_canvas = FigureCanvasTkAgg(self.enlarged_fig, master=popup)
            self.enlarged_canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')
            
            # Close behavior
            popup.protocol("WM_DELETE_WINDOW", lambda: close_enlarged(popup))
            
            # Mulicursor
            self.enlarged_cursor_h = self.enlarged_ax.axhline(y=self.cursor_h.get_ydata())
            self.enlarged_cursor_v = self.enlarged_ax.axvline(x=self.cursor_v.get_xdata())
            
            # Sync data
            self.update_enlarged_image()