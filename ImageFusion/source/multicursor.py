import matplotlib.pyplot as plt
from matplotlib.backend_tools import Cursors
import numpy as np

class MultiCursor:
    
    def __init__(self, images_by_views, controls):
        self.controls = controls
        self.figs = []
        self.axs = []
        self.cursor_hs = []
        self.cursor_vs = []
        self.image_views = []
        self.z_conv_factors = np.array([])
        self.y_conv_factors = np.array([])
        self.x_conv_factors = np.array([])
        
        for image_views in images_by_views:
            for view, image_view in enumerate(image_views):
                
                self.figs.append(image_view.fig)
                self.axs.append(image_view.ax)
                self.cursor_hs.append(image_view.cursor_h)
                self.cursor_vs.append(image_view.cursor_v)
                self.image_views.append(image_view)
                
                conv_factors = image_view.X.vxl_dims
                
                x_cf = conv_factors[2]
                y_cf = conv_factors[1]
                z_cf = conv_factors[0]
                
                self.x_conv_factors = np.append(self.x_conv_factors, x_cf)
                self.y_conv_factors = np.append(self.y_conv_factors, y_cf)
                self.z_conv_factors = np.append(self.z_conv_factors, z_cf)
                
                # image_view.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
                image_view.fig.canvas.mpl_connect('button_press_event', self.on_click)
                image_view.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
    
    def get_xyz_mm_from_xyz_indices(self, ind, x, y, z):        
        x_mm = x * self.x_conv_factors[ind]
        y_mm = y * self.y_conv_factors[ind]
        z_mm = z * self.z_conv_factors[ind]
        return x_mm, y_mm, z_mm
    
    def get_xyz_indices_from_xyz_xx(self, ind, x_mm, y_mm, z_mm):        
        x = np.floor(x_mm / self.x_conv_factors[ind]).astype(int)
        y = np.floor(y_mm / self.y_conv_factors[ind]).astype(int)
        z = np.floor(z_mm / self.z_conv_factors[ind]).astype(int)
        return x, y, z
    
    def get_slice_indicator(self, view, xyz):
        zyx = xyz[::-1]
        return zyx[view]
    
    def get_xyz_indices_from_event(self, event, ind):
        view = ind%3
        
        if view == 0:
            x = event.xdata
            y = event.ydata
            z = self.image_views[ind].X.slice_numbers[view]
    
        elif view == 1:
            x = event.xdata
            y = self.image_views[ind].X.slice_numbers[view]
            z = event.ydata
    
        else: # view == 2:
            x = self.image_views[ind].X.slice_numbers[view]
            y = event.xdata
            z = event.ydata
        
        return x, y, z
    
    def on_scroll(self, event):
        if event.inaxes in self.axs:
            ind = self.axs.index(event.inaxes)
            panel_ind = np.floor(ind/3).astype(int)
            view = ind%3
            v = self.image_views[ind].X.slice_numbers.copy()[view]
            if event.button == 'up':
                if v < self.image_views[ind].X.vxls_in_dim[view]-1:            
                    v = v - 1
            else: # event.button == 'down':
                if v > 0:    
                    v = v + 1
            self.controls[panel_ind].set_view_slice(view=view, slice_indicator=v, mode='by_number', from_mc=True)
            
    def on_move(self, event):
        ...
            
    def on_click(self, event):
        if event.button == 1 and event.inaxes in self.axs:
            ind = self.axs.index(event.inaxes)
            x, y, z = self.get_xyz_indices_from_event(event, ind)
            x_mm, y_mm, z_mm = self.get_xyz_mm_from_xyz_indices(ind, x, y, z)
            self.update_panel_slices(ind, x_mm, y_mm, z_mm)
            self.update_crosshairs()
                      
    def update_panel_slices(self, trigger_ind, x_mm, y_mm, z_mm):
        trigger_panel_ind = np.floor(trigger_ind/3).astype(int)
        for ind, _ in enumerate(self.axs):
            panel_ind = np.floor(ind/3).astype(int)
            if panel_ind == trigger_panel_ind:
                x, y, z = self.get_xyz_indices_from_xyz_xx(ind, x_mm, y_mm, z_mm)
                self.update_slice(ind, x, y, z)
                
    def update_slice(self, ind, x, y, z):
        view = ind%3
        panel_ind = np.floor(ind/3).astype(int)
        slice_indicator = self.get_slice_indicator(view, (x, y, z))
        self.controls[panel_ind].set_view_slice(view=view, slice_indicator=slice_indicator, mode='by_number', from_mc=True)
                
    def update_crosshairs(self):
        for ind, _ in enumerate(self.axs):
            view = ind%3
            panel_ind = np.floor(ind/3).astype(int)
            z = self.controls[panel_ind].views_slice_index[0].get()
            y = self.controls[panel_ind].views_slice_index[1].get()
            x = self.controls[panel_ind].views_slice_index[2].get()
            
            if view == 0:
                self.cursor_hs[ind].set_ydata(y)
                self.cursor_vs[ind].set_xdata(x)
                
            elif view == 1:
                self.cursor_hs[ind].set_ydata(z)
                self.cursor_vs[ind].set_xdata(x)
                
            else: # view == 2:
                self.cursor_hs[ind].set_ydata(z)
                self.cursor_vs[ind].set_xdata(y)
            
            self.image_views[ind].canvas.draw_idle()