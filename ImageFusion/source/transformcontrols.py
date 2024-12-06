import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import affine_transform

class TransformControls:
    def __init__(self, app, image_controls, parent_frame):
        self.app = app
        self.image_controls = image_controls
        self.parent_frame = parent_frame
        
        self.transform_points_LB = tk.Listbox(self.parent_frame) 
        self.transform_points_LB.pack(side=tk.TOP, anchor="nw")
        self.transform_points = []
        self.affine_matrix = []
        
        add_point_button = self.make_button(self.parent_frame, "Add Point", self.add_point)
        add_point_button.pack(anchor='nw', side='top', padx=5, pady=2)
        
        compute_transform_button = self.make_button(self.parent_frame, "Compute Transform", self.compute_transform)
        compute_transform_button.pack(anchor='nw', side='top', padx=5, pady=2)
        
        apply_transform_button = self.make_button(self.parent_frame, "Apply Transform", self.apply_transform)
        apply_transform_button.pack(anchor='nw', side='top', padx=5, pady=2)
    
    def make_button(self, parent, label, command):
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
        return button

    def add_point(self):
        point = [self.image_controls.view_controls.views_slice_index[0].get(),
                 self.image_controls.view_controls.views_slice_index[1].get(),
                 self.image_controls.view_controls.views_slice_index[2].get()]
        point_text = ", ".join(str(x) for x in point)
        self.transform_points.append(point)
        self.transform_points_LB.insert(tk.END, point_text)
        
    def compute_transform(self):
        if self.app.panel_1_controls.transform_controls.transform_points and self.app.panel_2_controls.transform_controls.transform_points:
        
            # Convert points to numpy arrays
            P1 = np.array(self.app.panel_1_controls.transform_controls.transform_points)
            P2 = np.array(self.app.panel_2_controls.transform_controls.transform_points)
            
            # Convert to mm
            P1 = P1 * self.app.X_CT.vxl_dims
            P2 = P2 * self.app.X_PET.vxl_dims
            
            if P1.shape == P2.shape:
            
                # Compute centroids
                centroid_P1 = np.mean(P1, axis=0)
                centroid_P2 = np.mean(P2, axis=0)
                
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
        
                self.affine_matrix = affine_matrix
    
    def apply_transform(self):
        def create_scale_matrix(voxel_dims):
            scale_factors = np.diag(1 / voxel_dims)
            scale_factors_4x4 = np.eye(4)
            scale_factors_4x4[:3, :3] = scale_factors
            return scale_factors_4x4
        
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
            self.app.refresh_graphics()