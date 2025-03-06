import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DualImageView:
    
    def __init__(self, app, parent, image_view_1, image_view_2):
        
        # Initialize
        self.app = app
        self.enlarged_flag = False
        self.image_view_1 = image_view_1
        self.image_view_2 = image_view_2
        self.view = self.image_view_1.view
        self.opacity = 0.5
        
        # Create a new figure and axis and display the image
        self.fig = plt.Figure() # figsize=(3, 3)
        self.ax = self.fig.add_subplot()
        
        self.image_1 = self.ax.imshow(self.image_view_1.image.get_array(),
                                      extent=self.image_view_1.get_extent(),
                                      origin='lower')
        
        self.image_2 = self.ax.imshow(self.image_view_2.image.get_array(),
                                      extent=self.image_view_2.get_extent(),
                                      alpha=self.opacity,
                                      origin='lower')
        
        # Set title and axes
        self.ax.set_title(self.image_view_1.ax.get_title())
        self.ax.set_xlabel(self.image_view_1.ax.get_xlabel())   
        self.ax.set_ylabel(self.image_view_1.ax.get_ylabel())
        
        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')

        # Bind spacebar event to open enlarged plot
        self.canvas.get_tk_widget().bind("<Button-2>", self.enlarge_plot)
        self.canvas.get_tk_widget().bind("<Button-3>", self.enlarge_plot)
        self.canvas.get_tk_widget().focus_set()  # Ensure it can capture key events
        
        self.update_data()
    
    def draw(self):
        self.canvas.draw_idle()
        
        if self.enlarged_flag:
            self.enlarged_canvas.draw_idle()
    
    def update_data(self):
        
        self.image_1.set_data(self.image_view_1.image.get_array())
        self.image_1.set_cmap(self.image_view_1.image.get_cmap())
        self.image_1.set_clim(self.image_view_1.image.get_clim())
        self.image_1.set_interpolation(self.image_view_1.image.get_interpolation())
        self.image_1.set_extent(self.image_view_1.image.get_extent())
        
        self.image_2.set_data(self.image_view_2.image.get_array())
        self.image_2.set_cmap(self.image_view_2.image.get_cmap())
        self.image_2.set_clim(self.image_view_2.image.get_clim())
        self.image_1.set_interpolation(self.image_view_2.image.get_interpolation())
        self.image_2.set_extent(self.image_view_2.image.get_extent())
        self.image_2.set_alpha(self.opacity)
        
        if self.enlarged_flag:
            self.update_enlarged_image()

    # --- Enlarged Plot --- #

    def update_enlarged_image(self):
        
        self.enlarged_image_1.set_data(self.image_1.get_array())
        self.enlarged_image_1.set_cmap(self.image_1.get_cmap())
        self.enlarged_image_1.set_clim(self.image_1.get_clim())
        self.enlarged_image_1.set_interpolation(self.image_1.get_interpolation())
        self.enlarged_image_1.set_extent(self.image_1.get_extent())
        
        self.enlarged_image_2.set_data(self.image_2.get_array())
        self.enlarged_image_2.set_cmap(self.image_2.get_cmap())
        self.enlarged_image_2.set_clim(self.image_2.get_clim())
        self.enlarged_image_2.set_interpolation(self.image_2.get_interpolation())
        self.enlarged_image_2.set_extent(self.image_2.get_extent())        
        self.enlarged_image_2.set_alpha(self.opacity)
 
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
            
            self.enlarged_image_1 = self.enlarged_ax.imshow(self.image_1.get_array(),
                                                            extent=self.image_1.get_extent())
            
            self.enlarged_image_2 = self.enlarged_ax.imshow(self.image_2.get_array(),
                                                            extent=self.image_2.get_extent(),
                                                            alpha=self.opacity)
            
            # Set title and axes
            self.enlarged_ax.set_title(self.image_view_1.ax.get_title())
            self.enlarged_ax.set_xlabel(self.image_view_1.ax.get_xlabel())   
            self.enlarged_ax.set_ylabel(self.image_view_1.ax.get_ylabel())
            
            # Adjust layout
            self.enlarged_fig.tight_layout()
            popup.transient(self.app) # set to be on top of the main window
                        
            # Create the canvas
            self.enlarged_canvas = FigureCanvasTkAgg(self.enlarged_fig, master=popup)
            self.enlarged_canvas.get_tk_widget().pack(padx=0, pady=0, expand=1, fill='both')
            
            # Close behavior
            popup.protocol("WM_DELETE_WINDOW", lambda: close_enlarged(popup))
            
            self.update_enlarged_image()