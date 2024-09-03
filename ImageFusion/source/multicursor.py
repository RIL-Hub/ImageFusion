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
                
                image_view.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
                image_view.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def get_xyz_mm_from_event(self, event, ind):
        x, y, z = self.get_xyz_from_event(event, ind)
        
        x_mm = x * self.x_conv_factors[ind]
        y_mm = y * self.y_conv_factors[ind]
        z_mm = z * self.z_conv_factors[ind]
        
        return x_mm, y_mm, z_mm
        
    def get_xyz_from_event(self, event, ind):
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
    
    def on_move(self, event):
        if event.inaxes in self.axs:
            
            ind = self.axs.index(event.inaxes)
            x_mm, y_mm, z_mm = self.get_xyz_mm_from_event(event, ind)
            
            self.update_crosshair(x_mm, y_mm, z_mm)
            
    def on_click(self, event):
        if event.button == 1 and event.inaxes in self.axs:
            
            ind = self.axs.index(event.inaxes)
            x_mm, y_mm, z_mm = self.get_xyz_mm_from_event(event, ind)
            
            self.update_slices(ind, x_mm, y_mm, z_mm)
    
    def update_crosshair(self, x_mm, y_mm, z_mm):
        xs = x_mm / self.x_conv_factors
        ys = y_mm / self.y_conv_factors
        zs = z_mm / self.z_conv_factors
        
        for ind, _ in enumerate(self.axs):
            view = ind%3
                        
            if view == 0:
                self.cursor_hs[ind].set_ydata(ys[ind])
                self.cursor_vs[ind].set_xdata(xs[ind])
                
            elif view == 1:
                self.cursor_hs[ind].set_ydata(zs[ind])
                self.cursor_vs[ind].set_xdata(xs[ind])
                
            else: # view == 2:
                self.cursor_hs[ind].set_ydata(zs[ind])
                self.cursor_vs[ind].set_xdata(ys[ind])
            
            self.image_views[ind].canvas.draw_idle()
    
    def update_slices(self, trigger_ind, x_mm, y_mm, z_mm):
        trigger_panel_ind = np.floor(trigger_ind/3).astype(int)
        
        for ind, _ in enumerate(self.axs):
            view = ind%3
            panel_ind = np.floor(ind/3).astype(int)
            controller = self.controls[panel_ind]
            
            if controller.linked_images.get() == 0:
                if panel_ind != trigger_panel_ind:
                    continue
            
            x = np.floor(x_mm / self.x_conv_factors[ind]).astype(int)
            y = np.floor(y_mm / self.y_conv_factors[ind]).astype(int)
            z = np.floor(z_mm / self.z_conv_factors[ind]).astype(int)
                        
            if view == 0:
                controller.set_view_slice(view, z, 'by_number', True)
                
            elif view == 1:
                controller.set_view_slice(view, y, 'by_number', True)
                
            else: # view == 2:
                controller.set_view_slice(view, x, 'by_number', True)
            
            controller.update_dual_view()
    
    def set_crosshair(self, trigger_view, v_mm):
        if trigger_view == 0:
            zs = v_mm / self.z_conv_factors
            for ind, _ in enumerate(self.axs):
                view = ind%3
                if view == 1 or view == 2: 
                    self.cursor_hs[ind].set_ydata(zs[ind])
        
        elif trigger_view == 1:
            ys = v_mm / self.y_conv_factors
            for ind, _ in enumerate(self.axs):
                view = ind%3
                if view == 0: 
                    self.cursor_hs[ind].set_ydata(ys[ind])
                elif view == 2:
                    self.cursor_vs[ind].set_xdata(ys[ind])
        
        else: # trigger_view == 2:
            xs = v_mm / self.x_conv_factors
            for ind, _ in enumerate(self.axs):
                view = ind%3
                if view == 0 or view == 1: 
                    self.cursor_vs[ind].set_xdata(xs[ind])
        
        for fig in self.figs:
            fig.canvas.draw_idle()