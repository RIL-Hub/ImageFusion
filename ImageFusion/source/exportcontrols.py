import tkinter as tk


class ExportControls:
    def __init__(self, app, image_controls, parent_frame):
        self.app = app
        self.image_controls = image_controls
        self.parent_frame = parent_frame
        
        options = [
            "Transverse",
            "Sagittal",
            "Coronal"
        ]
        self.export_view = tk.StringVar()
        self.make_export_dropdown("View", self.parent_frame, options, self.export_view)
        
        options = [
            "100",
            "300",
            "400",
            "600",
            "800",
            "1200"
        ]
        self.export_dpi = tk.StringVar()
        self.make_export_dropdown("DPI", self.parent_frame, options, self.export_dpi)
        
        options = [
            "png",
            "jpg",
            "svg",
            "eps"
        ]
        self.export_filetype = tk.StringVar()
        self.make_export_dropdown("Filetype", self.parent_frame, options, self.export_filetype)
        
        self.make_button(self.parent_frame, "Export", self.export)

    def make_export_dropdown(self, name, parent_frame, options, var):
        drop_frame = tk.Frame(parent_frame)
        drop_frame.pack(side='top', anchor='w')
        
        drop_label = tk.Label(drop_frame, text=name, width=7)
        drop_label.pack(side='left')
        var.set(options[0])
        drop_menu = tk.OptionMenu(drop_frame , var, *options)
        drop_menu.config(width=10, anchor='w')
        drop_menu.pack(side='left')
    
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
        button.pack(anchor='nw', side='top', padx=5, pady=2)
        return button
    
    def export(self):
        filename=self.export_view.get()
        dpi = int(self.export_dpi.get())
        extension = "." + self.export_filetype.get()
        
        filetype_dict = {
            ".png": "PNG files",
            ".jpg": "JPEG files",
            ".svg": "SVG files",
            ".eps": "EPS files",
            ".*": "All files"
        }
        
        # Build the filetypes list dynamically
        filetypes = [(filetype_dict[extension], f"*{extension}")]
        filetypes += [
            (desc, f"*{ext}") for ext, desc in filetype_dict.items() if ext != extension
        ]
        
        # Open file dialog for save location
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension = extension,
            filetypes=filetypes,
            initialfile=filename
        )
        
        if file_path:  # If the user didn't cancel
            view_dict = {"Transverse": 0, "Sagittal": 1, "Coronal": 2}
            self.image_controls.panel_views[view_dict[self.export_view.get()]].fig.savefig(file_path, dpi=dpi, bbox_inches='tight')