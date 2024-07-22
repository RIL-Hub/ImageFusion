import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tifffile import imread
import glob
import pydicom as dicom
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable

class App(tk.Tk):
    
    def __init__(self, title):
        # main setup
        super().__init__()
        self.title(title)
        self.geometry("1600x1000")
        self.minsize(800,400)
        
        # debug init
        data_path = 'D:/Code/Code_Research/Projects/CT2Attenuation/'

        # PET Images (stored as 3D .tifs)
        PET_path = 'nasir_recon-20240606T222236Z-003/'
        PET_image_path = data_path + PET_path + 'USCS_Orgain_Soil_4Cap_150uC_fud_12uCi_1hour_11_53am.E15.V0.5I50RAW.tif'
        file_properties = {"vxl_dim1_size": 0.5, "vxl_dim2_size": 0.5, "vxl_dim3_size": 0.5}
        self.X_PET = ImageData(PET_image_path, file_properties, image_type='PET')
        
        # CT Images (stored as series of .IMA files)
        CT_dir_path = '2024-05-23_EARTH_SHOT_CAP_TUBES_BENNETT-20240606T205140Z-001/'
        CT_images_path = data_path + CT_dir_path + 'DYNACT_HEAD_NAT_FILL_HU_NORMAL_180UM_VOXEL_109KV/'
        file_properties = {}
        self.X_CT = ImageData(CT_images_path, file_properties, image_type='CT')
        
        self.match_image_dims()
        self.X_PET.print_info()
        self.X_CT.print_info()
        
        # top menu
        self.config(menu=MenuBar(self))
        
        # scanner panels
        self.panel_1 = ScannerPanel(self)
        self.panel_2 = ScannerPanel(self)
        self.panel_3 = ScannerPanel(self)
        
        # image views
        self.image_1_view_1 = ImageView(self.panel_1.image_view_1, self.X_PET, view=0)
        self.image_1_view_2 = ImageView(self.panel_1.image_view_2, self.X_PET, view=1)
        self.image_1_view_3 = ImageView(self.panel_1.image_view_3, self.X_PET, view=2)
        self.image_1_views = [self.image_1_view_1, self.image_1_view_2, self.image_1_view_3]
        
        self.image_2_view_1 = ImageView(self.panel_2.image_view_1, self.X_CT, view=0)
        self.image_2_view_2 = ImageView(self.panel_2.image_view_2, self.X_CT, view=1)
        self.image_2_view_3 = ImageView(self.panel_2.image_view_3, self.X_CT, view=2)
        self.image_2_views = [self.image_2_view_1, self.image_2_view_2, self.image_2_view_3]
        
        self.image_3_view_1 = ImageView(self.panel_3.image_view_1, self.X_PET, view=0)
        self.image_3_view_2 = ImageView(self.panel_3.image_view_2, self.X_PET, view=1)
        self.image_3_view_3 = ImageView(self.panel_3.image_view_3, self.X_PET, view=2)
        self.image_3_views = [self.image_3_view_1, self.image_3_view_2, self.image_3_view_3]
        
        self.images_views = [self.image_1_views, self.image_2_views, self.image_3_views]
        
        # panel controls
        self.panel_1_controls = ImageControls(self.panel_1.image_controls, self, self.image_1_views, self.X_PET)
        self.panel_2_controls = ImageControls(self.panel_2.image_controls, self, self.image_2_views, self.X_CT)
        self.panel_3_controls = ImageControls(self.panel_3.image_controls, self, self.image_3_views, self.X_PET)

        # run
        self.mainloop()
        
    def match_image_dims(self):
        self.max_dims = [np.max([dim1, dims2]) for (dim1, dims2) in zip(self.X_PET.dims, self.X_CT.dims)]
        self.X_PET.pad_to_dims(self.max_dims)
        self.X_CT.pad_to_dims(self.max_dims)
        
        self.vxls_per_dim = [-1, -1, -1]
        self.vxl_length_per_dim = [-1, -1, -1]
        for i in range(3):
            if self.X_PET.vxls_in_dim[i] > self.X_CT.vxls_in_dim[i]:
                self.vxls_per_dim[i] = self.X_PET.vxls_in_dim[i]
                self.vxl_length_per_dim[i] = self.X_PET.vxl_dims[i]
            else:
                self.vxls_per_dim[i] = self.X_CT.vxls_in_dim[i]
                self.vxl_length_per_dim[i] = self.X_CT.vxl_dims[i]
    
    def refresh_graphics(self):
        self.image_1_view_1.canvas.draw()
        self.image_1_view_2.canvas.draw()
        self.image_1_view_3.canvas.draw()
        
        self.image_2_view_1.canvas.draw()
        self.image_2_view_2.canvas.draw()
        self.image_2_view_3.canvas.draw()
        
        self.image_3_view_1.canvas.draw()
        self.image_3_view_2.canvas.draw()
        self.image_3_view_3.canvas.draw()

class MenuBar(tk.Menu):
    def __init__(self, parent):
        super().__init__(parent)
        
        file_menu = tk.Menu(self, tearoff=0)
        file_menu.add_command(label="New", command=self.donothing)
        file_menu.add_command(label="Open", command=self.donothing)
        file_menu.add_command(label="Save", command=self.donothing)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.donothing)
        self.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(self, tearoff=0)
        help_menu.add_command(label="Help Index", command=self.donothing)
        help_menu.add_command(label="About...", command=self.donothing)
        help_menu.add_command(label="Help", command=self.donothing)
        self.add_cascade(label="Help", menu=help_menu)
        
    def donothing(self):
        pass

class DragDropMixIn:

    def make_draggable(self):
        self.dragging_state = False
        self.drag_zone.bind('<Enter>', self.on_start_hover)
        self.drag_zone.bind('<Leave>', self.on_end_hover)
        self.drag_zone.bind("<ButtonPress-1>", self.on_start)
        self.drag_zone.bind("<B1-Motion>", self.on_drag)
        self.drag_zone.bind("<ButtonRelease-1>", self.on_drop)
        self.drag_zone.configure(cursor="sb_v_double_arrow")

    def set_drag_colors(self):
        self.config(bg="gold")
        self.drag_zone.config(bg='goldenrod')
    
    def set_hover_colors(self):
        self.config(bg="cornflower blue")
        self.drag_zone.config(bg='royal blue')

    def on_start_hover(self, event):
        if not self.dragging_state:
            self.drag_zone.config(bg='gold')
    
    def on_end_hover(self, event):
        if not self.dragging_state:
            self.drag_zone.config(bg='light gray')

    def get_parent_before_toplevel(self, widget):
        while widget is not None and str(widget.master) != ".":
            widget = widget.master
        return widget
    
    def get_drag_target(self, event):
        cursor_x, cursor_y = event.widget.winfo_pointerxy()
        drag_target = event.widget.winfo_containing(cursor_x, cursor_y)
        drag_target = self.get_parent_before_toplevel(drag_target)
        return drag_target
    
    def get_drag_direction(self, event):
        _, cursor_y = event.widget.winfo_pointerxy()
        direction = -np.sign(cursor_y - self.drag_y_start)
        return direction
    
    def on_start(self, event):
        self.dragging_state = True
        _, self.drag_y_start = event.widget.winfo_pointerxy()
        self.set_drag_colors()
        self.drag_zone.configure(cursor="icon")
        self.last_drag_target = None

    def on_drag(self, event):
        direction = self.get_drag_direction(event)
        if direction > 0: self.drag_zone.configure(cursor="based_arrow_up")
        if direction < 0: self.drag_zone.configure(cursor="based_arrow_down")
        
        drag_target = self.get_drag_target(event)
        
        # trying to drag off window
        if drag_target is None:
            return 0
        
        # trying to drag onto self
        if drag_target == self:
            self.drag_zone.configure(cursor="icon")
            if self.last_drag_target:
                self.last_drag_target.set_neutral_colors()
            self.last_drag_target = None
            return 0
        
        # valid drag
        if self.last_drag_target:
            self.last_drag_target.set_neutral_colors()
        drag_target.set_hover_colors()
        self.last_drag_target = drag_target

        return 0

    def on_drop(self, event):
        self.dragging_state = False
        self.set_neutral_colors()
        self.drag_zone.configure(cursor="sb_v_double_arrow")
        
        if self.last_drag_target:
            self.last_drag_target.set_neutral_colors()
            
        drag_target = self.get_drag_target(event)
        direction = self.get_drag_direction(event)
        
        # trying to drag off window
        if drag_target is None:
            pass
        elif drag_target == self:
            self.drag_zone.config(bg='gold')
        elif direction > 0:
            self.pack(before=drag_target) 
        elif drag_target and direction < 0:
            self.pack(after=drag_target)

class ScannerPanel(DragDropMixIn, tk.Frame):
    def __init__(self, parent):
        # initialize parents
        tk.Frame.__init__(self, parent)
        DragDropMixIn.__init__(self)
        self.parent = parent
        
        self.pack(side='top', padx=2.5, pady=2.5, expand=1, fill='both')
        
        # drag widget
        self.drag_zone = tk.Frame(self, width=25, bd=1, relief=tk.RAISED)
        self.drag_zone.pack(side='left', padx=2.5, pady=2.5, expand=False, fill='both')
        self.make_draggable()
        
        # control widget
        self.image_controls = ttk.Notebook(self, width=250)
        self.image_controls.pack(side='left', padx=2.5, pady=2.5, expand=False, fill='both')

        # image widgets
        self.image_views = tk.Frame(self)
        self.image_views.pack(side='left', padx=2.5, pady=2.5, expand=1, fill='both')
        self.image_view_1 = tk.Frame(self.image_views, bg='blue')
        self.image_view_2 = tk.Frame(self.image_views, bg='blue')
        self.image_view_3 = tk.Frame(self.image_views, bg='blue')
        self.image_view_1.pack(padx=2.5, pady=2.5, side='left', expand=True, fill='both')
        self.image_view_2.pack(padx=2.5, pady=2.5, side='left', expand=True, fill='both')
        self.image_view_3.pack(padx=2.5, pady=2.5, side='left', expand=True, fill='both')
        
        # set colors
        self.set_neutral_colors()

    def set_neutral_colors(self):
        self.config(bg="light gray")
        self.drag_zone.config(bg='light gray')
        self.image_views.config(bg='red')
           
class ImageView:
    def __init__(self, parent, X, view):
        self.fig = plt.Figure()
        self.fig.set_figheight(3)
        self.fig.set_figwidth(3)
        self.ax = self.fig.add_subplot()
        
        self.X_max = X.max_intensity
        self.view = view
        self.threshold = 1
        self.thresh_min = 0
        self.thresh_max = self.X_max
        self.X = X
        self.X.set_slice(100, view)
        
        self.image = self.ax.imshow(self.X.slice, vmin=0, vmax=self.X.get_slice_max(), cmap='gist_yarg', interpolation='none')
        
        # Color Bar
        divider_PET = make_axes_locatable(self.ax)
        cb_cax = divider_PET.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(self.image, cax=cb_cax)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(padx=2.5, pady=2.5, expand=1, fill='both')
    
    def set_slice(self, slice_number):
        self.X.set_slice(slice_number, self.view)
        self.update_data()
    
    def update_data(self):
        self.image.set_data(self.X.slice)
        # self.image.set_clim(vmin=0, vmax=self.X.get_slice_max())
        self.cbar.update_normal(self.image)
        
        # self.canvas.flush_events()
        self.canvas.draw()

class ImageData:
    def __init__(self, file_path, file_properties, image_type):
        self.file_path = file_path
        self.file_properties = file_properties
        self.image_type = image_type
        
        if self.image_type == 'PET':
            self.init_PET_image()
        if self.image_type == 'CT':
            self.init_CT_image()
        if self.image_type == 'dual':
            self.init_dual_image()
        
        self.X = self.original_X.copy()
        self.max_intensity = np.amax(self.X)
        self.set_slice(0, 0)
        
        self.print_info()
       
    def print_info(self):
        print(f'{self.image_type}')
        print(f'Image Size: {self.vxls_in_dim[0]} x {self.vxls_in_dim[1]} x {self.vxls_in_dim[2]} vxls')
        print(f'Image Dims: [{self.dims[0]}, {self.dims[1]}, {self.dims[2]}] mm')
        print(f'Voxel Dims: [{self.vxl_dims[0]}, {self.vxl_dims[1]}, {self.vxl_dims[2]}] mm')
        print('')

    def init_PET_image(self):
        self.original_X = np.flip(imread(self.file_path), axis=0)
        
        vxl_dim1 = self.file_properties['vxl_dim1_size']
        vxl_dim2 = self.file_properties['vxl_dim2_size']
        vxl_dim3 = self.file_properties['vxl_dim3_size']
        self.vxl_dims = [vxl_dim1, vxl_dim2, vxl_dim3]
        
        vxls_in_dim1 = int(self.original_X.shape[0])
        vxls_in_dim2 = int(self.original_X.shape[1])
        vxls_in_dim3 = int(self.original_X.shape[2])
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        
        self.set_dims()
        
    def init_CT_image(self):
        plots = []
        file_names = []
        vxls_in_dim3 = 0
        for idx, file_name in enumerate(glob.glob(self.file_path + '*.IMA')):
            ds = dicom.dcmread(file_name)
            pix = ds.pixel_array
            plots.append(pix)
            file_name = file_name.split('\\')[-1]
            file_names.append(file_name)  # Save the filename
            vxls_in_dim3 = vxls_in_dim3 + 1
            if idx == 0:
                vxl_dim1 = float(ds.PixelSpacing[0])
                vxl_dim2 = float(ds.PixelSpacing[1])
                vxl_dim3 = float(ds.SliceThickness)
                vxls_in_dim1 = int(ds.Rows)
                vxls_in_dim2 = int(ds.Columns)
        
        self.vxl_dims = [vxl_dim1, vxl_dim2, vxl_dim3]
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        self.original_X = np.dstack(plots)
        self.set_dims()
        
    def init_Dual_image(self):
        ...
    
    def set_slice(self, slice_number, view):
        slice_number = int(slice_number)
        if view == 0: self.slice = self.X[slice_number, :, :]
        if view == 1: self.slice = self.X[:, slice_number, :]
        if view == 2: self.slice = self.X[:, :, slice_number]
    
    def get_slice_max(self):
        return np.amax(self.slice)
    
    def set_slice_by_measure(self, slice_distance, view):
        slice_number = int(np.floor(np.divide(slice_distance, self.vxl_dims[view])))
        self.set_slice(slice_number, view)
        
    def pad_to_dims(self, target_dims):
        for i, (X_dim, T_dim) in enumerate(zip(self.dims, target_dims)):
            if X_dim < T_dim:
                voxels_to_add = int(round( (T_dim - X_dim)/self.vxl_dims[i] ))
                padding_width = [[0, 0], [0, 0], [0, 0]]
                padding_width[i][0] = voxels_to_add // 2
                padding_width[i][1] = voxels_to_add // 2
                self.X = np.pad(self.X, padding_width)
            
        self.set_vxls_in_dim()
    
    def set_vxls_in_dim(self):
        vxls_in_dim1 = int(self.X.shape[0])
        vxls_in_dim2 = int(self.X.shape[1])
        vxls_in_dim3 = int(self.X.shape[2])
        self.vxls_in_dim = [vxls_in_dim1, vxls_in_dim2, vxls_in_dim3]
        self.set_dims()
    
    def set_dims(self):
        self.dims = np.multiply(self.vxl_dims, self.vxls_in_dim)
             
class ImageControls:
    def __init__(self, parent_frame, app, panel_views, image):
        self.app = app
        self.panel_views = panel_views
        self.image = image
        self.views_slice_index = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        self.time_index = 0
        
        # view controls
        self.tab_view = ttk.Frame(parent_frame)
        parent_frame.add(self.tab_view, text='View')
        
        slice_controls = tk.Frame(self.tab_view, bd=1, relief=tk.SUNKEN)
        slice_controls.pack(anchor='n', side='left', padx=5, pady=2)
        
        slice_sliders = tk.Frame(slice_controls)
        self.slider_view_1 = self.make_slider(slice_sliders, view=0, name='V1')
        self.slider_view_2 = self.make_slider(slice_sliders, view=1, name='V2')
        self.slider_view_3 = self.make_slider(slice_sliders, view=2, name='V3')
        # self.slider_time   = self.make_slider(slice_sliders, variable=self.time_index, command=self.set_time_slice, name='T', bounds=[0, 10])
        slice_sliders.pack(side='top')
        
        # intensity_controls = tk.Frame(self.tab_view, bd=1, relief=tk.SUNKEN)
        # intensity_controls.pack(side='left', padx=5, pady=2)
        # self.intensity_slider = self.make_slider(intensity_controls, 0, self.set_intensity, 'I', bounds=[0, 100])
        
        self.make_linked_images(slice_controls)
        
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
                self.app.panel_3_controls.linked_images.set(1)
                
                self.set_view_slice(view=0, slice_number=self.views_slice_index[0].get()) 
                self.set_view_slice(view=1, slice_number=self.views_slice_index[1].get()) 
                self.set_view_slice(view=2, slice_number=self.views_slice_index[2].get()) 
            else:
                self.app.panel_1_controls.linked_images.set(0)
                self.app.panel_2_controls.linked_images.set(0)
                self.app.panel_3_controls.linked_images.set(0)
        
        self.linked_images = tk.IntVar(value=1)
        self.linked_images_checkbutton = tk.Checkbutton(parent, text="Link Images", onvalue=1, offvalue=0,
                                                        variable=self.linked_images, command=checkbutton_command)
        self.linked_images_checkbutton.pack(anchor='w', side='bottom')
    
    def make_slider(self, parent_frame, view, name):
        def slider_command(slider_value):
            slider_value = int(slider_value)
            slider_showvalue.config(text=(slider_value+1))
            self.set_view_slice(view, slider_value)
            
        slider_frame = tk.Frame(parent_frame)
        slider_frame.pack(side='left')
        
        slider = tk.Scale(slider_frame, variable=self.views_slice_index[view],
                          command=slider_command,
                          showvalue=False,
                          from_=0, to=self.image.vxls_in_dim[view]-1,
                          width=10, length=200, orient='vertical')
        slider_showvalue = tk.Label(slider_frame, text=slider.get())
        slider_label = tk.Label(slider_frame, text=name, width=3)
        
        slider_showvalue.pack(side='top')
        slider.pack(side='top')
        slider_label.pack(side='top')
        
        slider.set(99)
        
        return slider
    
    def set_self_view_slice(self, view, slice_number):
        self.views_slice_index[view].set(slice_number)
        self.panel_views[view].set_slice(slice_number)
        # TODO: update slide label based on position
        # TODO: fix index out of bound error for maxing out CT image sliders
    
    def slice_percent_to_slice_number(self, view, slice_percent):
        return int(np.round(slice_percent * self.image.vxls_in_dim[view]))
    
    def set_self_view_slice_percent(self, view, slice_percent):
        slice_number = self.slice_percent_to_slice_number(view, slice_percent)
        self.set_self_view_slice(view, slice_number)
    
    def set_view_slice(self, view, slice_number):
        if self.linked_images.get() == 1:
            slice_percent_of_dim = slice_number / self.image.vxls_in_dim[view]
            self.app.panel_1_controls.set_self_view_slice_percent(view, slice_percent_of_dim)
            self.app.panel_2_controls.set_self_view_slice_percent(view, slice_percent_of_dim)
            self.app.panel_3_controls.set_self_view_slice_percent(view, slice_percent_of_dim)
        else:
            self.set_self_view_slice(view, slice_number)
        
        slice_percent_of_dim = slice_number / self.image.vxls_in_dim[view]

    def set_time_slice(self, slice_number):
        ...
    
    def set_intensity(self, intensity):
        ...

def main() -> int:
    App("ImageFusion3D")
    return 0

if __name__ == "__main__":
    main()