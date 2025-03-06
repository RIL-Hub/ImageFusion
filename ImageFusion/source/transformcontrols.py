import numpy as np
import tkinter as tk
from scipy.ndimage import affine_transform
from numba import jit
import threading
from PIL import Image, ImageTk, ImageSequence
from functools import partial

@jit(nopython=True)
def compute_transform(P1, P2):
    
    # Compute centroids
    centroid_P1 = np.array([P1[:, i].mean() for i in range(P1.shape[1])])
    centroid_P2 = np.array([P2[:, i].mean() for i in range(P2.shape[1])])
    
    # Centralize points
    P1_centered = P1 - centroid_P1
    P2_centered = P2 - centroid_P2
    
    # Compute covariance matrix
    H = P1_centered.T @ P2_centered
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation matrix (det(R) should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_P2 - R @ centroid_P1
    
    # Create the affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R
    affine_matrix[:3, 3] = t

    return affine_matrix

class TransformControls:
    
    def __init__(self, app, image_data, image_controls, parent_frame):
        
        self.app = app
        self.image_data = image_data
        self.image_controls = image_controls
        self.parent_frame = parent_frame
        self.good_transform = False
        
        # --- Simple Transformation Controls --- #
        
        simple_frame = tk.Frame(parent_frame, bd=1, relief=tk.SUNKEN)
        simple_frame.pack(side='top', anchor='nw', padx=5, pady=5, fill="x")
        tk.Label(simple_frame, text="Simple Transformation").pack(side='top', anchor='n')
        
        self.make_simple_controls(simple_frame, 0)
        self.make_simple_controls(simple_frame, 1)
        self.make_simple_controls(simple_frame, 2)
                
        # --- Affine Transformation Controls --- #
        
        affine_frame = tk.Frame(parent_frame, bd=1, relief=tk.SUNKEN)
        affine_frame.pack(side='top', anchor='nw', padx=5, pady=5, fill="x")
        tk.Label(affine_frame, text="Affine Transformation").pack(side='top', anchor='n')
        
        self.transform_points_LB = tk.Listbox(affine_frame, width=20)
        self.transform_points_LB.pack(side='left', anchor="nw", padx=5, pady=5)
        self.transform_points = []
        self.affine_matrix = []
        
        affine_frame_right = tk.Frame(affine_frame)
        affine_frame_right.pack(side='left', anchor='nw', padx=0, pady=5)
        
        tk.Label(affine_frame_right, text="Reference Points").pack(side='top', anchor='n')
        reference_points_frame = tk.Frame(affine_frame_right)
        reference_points_frame.pack(side='top', anchor='nw', padx=0, pady=5)
        self.make_button(reference_points_frame, "Add", self.add_point).pack(anchor='w', side='left', padx=2.5, pady=2.5)
        self.make_button(reference_points_frame, "Remove", self.remove_point).pack(anchor='w', side='left', padx=2.5, pady=2.5)
        
        tk.Label(affine_frame_right, text="Trasnsform").pack(side='top', anchor='n')
        transform_frame = tk.Frame(affine_frame_right)
        transform_frame.pack(side='top', anchor='nw', padx=0, pady=5)
        self.make_button(transform_frame, "Compute", self.compute_transform_wrapper).pack(anchor='w', side='left', padx=2.5, pady=2.5)
        self.make_button(transform_frame, "Apply", self.apply_transform).pack(anchor='w', side='left', padx=2.5, pady=2.5)
        
        # --- Reset Transforms --- #
        
        self.make_button(self.parent_frame, "Reset Transforms", self.reset_transforms).pack(anchor='nw', side='top', padx=2.5, pady=2.5)
        
        
        # --- Initialize JIT Functions --- #
        
        dummy_points = np.random.rand(3,3)
        compute_transform(dummy_points, dummy_points)
        
    # --- Make Functions --- #
        
    def make_button(self, parent, label, command, width=None):
        button = tk.Button(
            parent, 
            text=label, 
            command=command,
            # activebackground="blue", 
            # activeforeground="white",
            anchor="center",
            bd=1,
            # bg="lightgray",
            cursor="hand2",
            # disabledforeground="gray",
            fg="black",
            font=("Arial", 8),
            # height=1,
            highlightbackground="black",
            highlightcolor="green",
            highlightthickness=2,
            justify="center",
            overrelief="raised",
            # padx=2,
            # pady=5,
            # width=15,
            wraplength=100)
        
        if width:
            button.config(width=width)
            width=None
            
        return button

    def make_simple_controls(self, parent_frame, view):
        titles = ['Transeverse', 'Coronal', 'Sagittal']
        title = titles[view]
        
        pad_x = 1
        pad_y = 1
        w = 3
        
        view_frame = tk.Frame(parent_frame)
        view_frame.pack(side='left', anchor='w', padx=pad_x, pady=5, fill='x', expand=True)
        tk.Label(view_frame, text=title).pack(side='top', anchor='n')
        button_frame = tk.Frame(view_frame)
        button_frame.pack(side='top', anchor='n', padx=pad_x, pady=5)
        
        rotate_clockwise = partial(self.rotate, view, 'clockwise')
        rotate_anticlockwise = partial(self.rotate, view, 'anticlockwise')
        invert_x = partial(self.invert, view, 'x')
        invert_y = partial(self.invert, view, 'y')
        
        cw = self.make_button(button_frame, "⟳", rotate_clockwise, width=w)
        aw = self.make_button(button_frame, "⟲", rotate_anticlockwise, width=w)
        ud = self.make_button(button_frame, "⇅", invert_y, width=w)
        lr = self.make_button(button_frame, "⇄", invert_x, width=w)
        
        cw.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky='N')
        aw.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky='N')
        ud.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky='N')
        lr.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky='N')
    
    # --- Simple Transforms --- #
    
    def rotate(self, view, direction):
        
        def rotate_slice_indices(x_view, y_view, z_view):
            
            # current slice indices
            slice_0_index = self.image_controls.view_controls.views_slice_index[TRANSVERSE].get()
            slice_1_index = self.image_controls.view_controls.views_slice_index[CORONAL].get()
            slice_2_index = self.image_controls.view_controls.views_slice_index[SAGITTAL].get()
            
            x_slice_index = self.image_controls.view_controls.views_slice_index[x_view].get()
            y_slice_index = self.image_controls.view_controls.views_slice_index[y_view].get()
            x = self.image_controls.view_controls.image_views[x_view].get_mm_from_slice_number(x_slice_index)
            y = self.image_controls.view_controls.image_views[y_view].get_mm_from_slice_number(y_slice_index)

            if direction == 'clockwise':
                t = y
                y = -x
                x = t
            else: # direction == 'anticlockwise':
                t = y
                y = x
                x = -t

            if z_view == 0:
                # x -> y = sagittal (2)
                # y -> x = coronal (1)
                slice_1_index = self.image_controls.view_controls.image_views[SAGITTAL].get_slice_number_from_mm(y)
                slice_2_index = self.image_controls.view_controls.image_views[CORONAL].get_slice_number_from_mm(x)
                
            elif z_view == 1:
                # x -> y = sagittal (2)
                # y -> x = transverse (0)
                slice_0_index = self.image_controls.view_controls.image_views[SAGITTAL].get_slice_number_from_mm(y)
                slice_2_index = self.image_controls.view_controls.image_views[TRANSVERSE].get_slice_number_from_mm(x)
                
            else: # z_view == 2:
                # x -> y coronal (1)
                # y -> x transverse (0)
                slice_0_index = self.image_controls.view_controls.image_views[CORONAL].get_slice_number_from_mm(y)
                slice_1_index = self.image_controls.view_controls.image_views[TRANSVERSE].get_slice_number_from_mm(x)
            
            self.image_controls.view_controls.views_slice_index[0].set(slice_0_index)
            self.image_controls.view_controls.views_slice_index[1].set(slice_1_index)
            self.image_controls.view_controls.views_slice_index[2].set(slice_2_index)
            
        TRANSVERSE = 0
        CORONAL = 1
        SAGITTAL = 2
        
        self.image_data.rotate_view(view, direction)
        
        # rearrange slice numbers        
        if view == 0: # transeverse
            # x is sagittal (2), y is coronal (1)
            rotate_slice_indices(SAGITTAL, CORONAL, TRANSVERSE)
        
        elif view == 1: # coronal
            # x is sagittal (2), y is transverse (0)
            rotate_slice_indices(SAGITTAL, TRANSVERSE, CORONAL)
            
        else: # view == 2: # sagittal
            # x is coronal (1), y is transverse (0)
            rotate_slice_indices(CORONAL, TRANSVERSE, SAGITTAL)
        
        # now fix all the "immutables" and refresh
        for image_view in self.image_controls.image_views:
            
            # rearrange extent
            extent = image_view.extent
            temp_0 = extent[0]
            temp_1 = extent[1]
            extent[0] = extent[2]
            extent[1] = extent[3]
            extent[2] = temp_0
            extent[3] = temp_1

            # update image
            image_view.extent = extent
            image_view.reload_slice()
            image_view.update_data()
        
        self.app.refresh_graphics()
        
    def invert(self, view, direction):
        if view == 0: # transeverse
            # x is sagittal (2), y is coronal (1)
            target_view = 2 if direction == 'x' else 1
        elif view == 1: # coronal
            # x is sagittal (2), y is transverse (0)
            target_view = 2 if direction == 'x' else 0
        else: # view == 2: # sagittal
            # x is coronal (1), y is transverse (0)
            target_view = 1 if direction == 'x' else 0
        
        self.image_data.invert_view(target_view)
        
        # update data and graphics
        self.app.reload_slices()
        self.app.refresh_data()
        self.app.refresh_graphics()
    
    def reset_transforms(self):
        self.image_data.X = self.image_data.original_X.copy()
        self.image_data.preload_slices()
        self.image_data.refresh_characteristics()
        
        self.image_controls.view_controls.set_view_slice(0, 0, 'by_mm', original_call=False)
        self.image_controls.view_controls.set_view_slice(1, 0, 'by_mm', original_call=False)
        self.image_controls.view_controls.set_view_slice(2, 0, 'by_mm', original_call=False)
        
        # update data and graphics
        self.app.reload_slices()
        self.app.refresh_data()
        self.app.refresh_graphics()
    
    # --- Affine Transform --- #
    
    def add_point(self):
        point = self.image_data.slice_mm.copy()
        point_text = ", ".join(f"{x:.2f}" for x in point)
        self.transform_points.append(point)
        self.transform_points_LB.insert(tk.END, point_text)
      
    def remove_point(self):
        if self.transform_points_LB.curselection():
            cursor_selection =self.transform_points_LB.curselection()[0]
            del self.transform_points[cursor_selection]
            self.transform_points_LB.delete(cursor_selection)
      
    def compute_transform_wrapper(self):
        if self.app.panel_1_controls.transform_controls.transform_points and self.app.panel_2_controls.transform_controls.transform_points:
        
            # Convert points to numpy arrays
            P1 = np.array(self.app.panel_1_controls.transform_controls.transform_points)
            P2 = np.array(self.app.panel_2_controls.transform_controls.transform_points)
            
            # Convert to mm
            P1 = P1 * self.app.X_CT.vxl_dims
            P2 = P2 * self.app.X_PET.vxl_dims
            
            if P1.shape == P2.shape:
                self.affine_matrix = compute_transform(P1, P2)
                self.good_transform = True
    
    def apply_transform(self):
        
        def create_scale_matrix(voxel_dims):
            scale_factors = np.diag(1 / voxel_dims)
            scale_factors_4x4 = np.eye(4)
            scale_factors_4x4[:3, :3] = scale_factors
            return scale_factors_4x4
    
        def run_transformation():
            if len(self.affine_matrix) > 0:
                inverse_affine = np.linalg.inv(self.affine_matrix)
                
                # Scaling matrices
                scale_factors_4x4 = create_scale_matrix(np.array(self.app.X_CT.vxl_dims))
                inverse_scale_factors_4x4 = create_scale_matrix(1 / np.array(self.app.X_CT.vxl_dims))
    
                # Convert affine matrix from mm space to voxel space
                voxel_affine = scale_factors_4x4 @ inverse_affine @ inverse_scale_factors_4x4
    
                # Extract rotation and translation
                rotation = voxel_affine[:3, :3]
                translation = voxel_affine[:3, 3]
    
                # Apply the affine transformation to the image
                self.app.X_CT.X = affine_transform(self.app.X_CT.X, rotation, offset=translation, order=1)
                self.app.X_CT.preload_slices()
                
                # Refresh UI graphics
                self.app.reload_slices()
                self.app.refresh_data()
                self.app.refresh_graphics()
                
                # Hide the loading screen once done
                self.app.after(0, hide_loading)
    
        def show_loading():
            self.app.update_idletasks()  # Ensure correct window size before creating overlay

            # Get correct window dimensions
            root_x = self.app.winfo_rootx()
            root_y = self.app.winfo_rooty()
            width = self.app.winfo_width()
            height = self.app.winfo_height()
        
            self.loading_overlay = tk.Toplevel(self.app)
            self.loading_overlay.attributes("-topmost", True)
            self.loading_overlay.geometry(f"{width}x{height}+{root_x}+{root_y}")            
            self.loading_overlay.configure(bg="white")
            self.loading_overlay.attributes("-alpha", 0.8)  # Adjust opacity (0.0 - 1.0)
            self.loading_overlay.overrideredirect(True)  # Remove window borders
            self.loading_overlay.resizable(False, False)  # Disable resizing
    
            # Center the loading GIF
            self.loading_label = tk.Label(self.loading_overlay, bg=None, bd=0, highlightthickness=0)
            self.loading_label.pack(expand=True)
    
            # Load the GIF
            self.loading_img = Image.open("images/loading.gif")
            self.loading_frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(self.loading_img)]
            self.loading_index = 0
    
            def update_gif():
                self.loading_index = (self.loading_index + 1) % len(self.loading_frames)
                self.loading_label.config(image=self.loading_frames[self.loading_index])
                self.loading_overlay.after(100, update_gif)  # Adjust speed if needed
    
            update_gif()
    
        def hide_loading():
            self.loading_overlay.destroy()
    
        if self.good_transform:
            
            # Show the loading overlay
            show_loading()
        
            # Run affine_transform in a separate thread
            threading.Thread(target=run_transformation, daemon=True).start()