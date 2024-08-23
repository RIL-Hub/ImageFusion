import matplotlib.pyplot as plt
from matplotlib.backend_tools import Cursors

class MultiCursor:
    
    def __init__(self, images_by_views):
                
        self.axs = []
        self.cursor_hs = []
        self.cursor_vs = []
        self.image_views = []
        self.row_conv_factors = []
        self.col_conv_factors = []
        
        for image_views in images_by_views:
            for view, image_view in enumerate(image_views):
                
                self.axs.append(image_view.ax)
                self.cursor_hs.append(image_view.cursor_h)
                self.cursor_vs.append(image_view.cursor_v)
                self.image_views.append(image_view)
                
                conv_factors = image_view.X.vxl_dims
                if view == 0:
                    row_cf = conv_factors[1]
                    col_cf = conv_factors[2]
                elif view == 1:
                    row_cf = conv_factors[0]
                    col_cf = conv_factors[2]
                else: # view == 2:
                    row_cf = conv_factors[0]
                    col_cf = conv_factors[1]
                
                self.row_conv_factors.append(row_cf)
                self.col_conv_factors.append(col_cf)
                
                image_view.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
    
    def on_move(self, event):
        if event.inaxes in self.axs:
            ind = self.axs.index(event.inaxes)
            x_mm = event.xdata * self.col_conv_factors[ind]
            y_mm = event.ydata * self.row_conv_factors[ind]
            self.update_crosshair(x_mm, y_mm)
    
    def update_crosshair(self, x_mm, y_mm):
        for ind, _ in enumerate(self.axs):
            x = x_mm / self.col_conv_factors[ind]
            y = y_mm / self.row_conv_factors[ind]
        
            self.cursor_hs[ind].set_ydata([y])
            self.cursor_vs[ind].set_xdata([x])
            self.image_views[ind].canvas.draw()