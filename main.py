import tkinter as tk
from tkinter import filedialog
from image_viewer import ImageViewer
 
root = tk.Tk()
root.title("Auto Annotation for Custom Data Segmentation Task Using SAM")
 
image_viewer = ImageViewer(root)
 
root.mainloop() 

 