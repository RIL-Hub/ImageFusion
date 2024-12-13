# import matplotlib.pyplot as plt
# import os

# def countlines(start, lines=0, header=True, begin_start=None):
#     if header:
#         print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
#         print('{:->11}|{:->11}|{:->20}'.format('', '', ''))

#     for thing in os.listdir(start):
#         thing = os.path.join(start, thing)
#         if os.path.isfile(thing):
#             if thing.endswith('.py'):
#                 with open(thing, 'r') as f:
#                     newlines = f.readlines()
#                     newlines = len(newlines)
#                     lines += newlines

#                     if begin_start is not None:
#                         reldir_of_thing = '.' + thing.replace(begin_start, '')
#                     else:
#                         reldir_of_thing = '.' + thing.replace(start, '')

#                     print('{:>10} |{:>10} | {:<20}'.format(
#                             newlines, lines, reldir_of_thing))


#     for thing in os.listdir(start):
#         thing = os.path.join(start, thing)
#         if os.path.isdir(thing):
#             lines = countlines(thing, lines, header=False, begin_start=start)

#     return lines

# def main():
#     target = 'D:\Code\Code_Research\Projects\ImageFusion\ImageFusion'
#     countlines(target)
    
# if __name__=="__main__":
#     main()

import tkinter as tk
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the main Tkinter window
root = tk.Tk()
root.title("Colormap Selector")

# List of colormaps from Matplotlib
colormaps = [
    "gist_yarg",
    "gist_gray",
    "inferno",
    "viridis",
    "hot",
    "jet"
]  # Sorted list of available colormaps
selected_colormap = tk.StringVar(value="viridis")  # Default colormap

# Function to handle colormap selection
def set_colormap(cmap):
    selected_colormap.set(cmap)
    print(f"Selected colormap: {cmap}")
    update_image_colormap()  # Apply the selected colormap to the image

# Function to update the image with the selected colormap
def update_image_colormap():
    # Generate a random image for demonstration purposes
    data = np.random.rand(10, 10)
    
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Display the image with the selected colormap
    cax = ax.imshow(data, cmap=selected_colormap.get())
    
    # Add color bar
    fig.colorbar(cax)

    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()



# Frame for image
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Update the image initially
update_image_colormap()

# Frame to simulate the dropdown
menu_frame = tk.Frame(root, padx=10, pady=10)
menu_frame.pack()

# Frame to hold the radio buttons for colormap selection
radio_button_frame = tk.Frame(menu_frame)

# Function to toggle the visibility of the radio buttons
def toggle_colormap_menu():
    if radio_button_frame.winfo_ismapped():
        radio_button_frame.pack_forget()
    else:
        show_colormap_options()  # Show the radio buttons
        radio_button_frame.pack(pady=(0, 10))

# Function to create radio buttons for colormap selection
def show_colormap_options():
    for widget in radio_button_frame.winfo_children():
        widget.destroy()  # Clear existing widgets

    # Create a radio button for each colormap
    for cmap in colormaps:
        rb = tk.Radiobutton(
            radio_button_frame,
            text=cmap,
            variable=selected_colormap,
            value=cmap,
            command=lambda: set_colormap(selected_colormap.get()),
            anchor="w",
        )
        rb.pack(fill="x")

# Button to open the radio button menu
colormap_button = tk.Button(
    menu_frame,
    text="Select Color Map",
    command=toggle_colormap_menu,
    relief="raised",
)
colormap_button.pack()

# Start the Tkinter main loop
root.mainloop()
