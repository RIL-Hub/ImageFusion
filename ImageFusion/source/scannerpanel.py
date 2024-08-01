import numpy as np
import tkinter as tk
from tkinter import ttk


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