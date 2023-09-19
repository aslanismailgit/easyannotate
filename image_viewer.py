import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from buttons import TopButtons, LeftButtons, RightButtons, BottomButtons

import numpy as np
import cv2
import os
import json
import shutil

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

from segment_anything import sam_model_registry, SamPredictor


device = "cuda"

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.width = 600
        self.height = 400
        self.zoom_factor = 1.0
        self.image = None
        self.file_path = None
        self.image_height = None
        self.image_width = None
        self.x_position = 0
        self.y_position = 0
        self.clicked_x_original = None
        self.clicked_y_original = None
        self.xyxy = []
        self.xyxy4draw = []
        self.masked_points = []

        self.model_type = None 
        self.sam_checkpoint = None
        self.predictor = None
        self.image_embedding = None
        self.masks = None
        self.scores = None
        self.logits = None
        self.mask = None

        self.class_names = []

        self.create_ui()

    def create_project_folder(self):
        if not os.path.exists("Projects"):
            os.mkdir("Projects") 

    def create_ui(self):
        self.create_project_folder()
        # Create frames for buttons
        button_frame_top = tk.Frame(self.root)
        button_frame_top.pack(side=tk.TOP, pady=10)

        button_frame_left = tk.Frame(self.root)
        button_frame_left.pack(side=tk.LEFT, padx=10)

        button_frame_right = tk.Frame(self.root)
        button_frame_right.pack(side=tk.RIGHT, padx=10)

        button_frame_bottom = tk.Frame(self.root)
        button_frame_bottom.pack(side=tk.BOTTOM, padx=10)

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()

        self.reset_button_frame = tk.Frame(self.root)
        self.reset_button_frame.pack(side=tk.BOTTOM)

        self.top_buttons = TopButtons(button_frame_top, self)
        self.left_buttons = LeftButtons(button_frame_left, self)
        self.right_buttons = RightButtons(button_frame_right, self)
        self.bottom_buttons = BottomButtons(button_frame_bottom, self)

        # Bind the mousewheel event to the move_x and move_y functions
        self.root.bind("<MouseWheel>", self.move_x)
        self.root.bind("<Shift-MouseWheel>", self.move_y)

        # Bind the canvas click event to the capture_click function
        self.canvas.bind("<Button-1>", self.capture_click)

        self.canvas.bind("<Motion>", self.on_mouse_move)

    def reset_variables(self):
        # Reset variables
        self.zoom_factor = 1.0
        self.image = None
        self.x_position = 0
        self.y_position = 0
        self.clicked_x_original = None
        self.clicked_y_original = None
        self.canvas.delete("all")

    def load_image(self):
        self.reset_variables()
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
                                                    initialdir=".")
        if self.file_path:
            img = Image.open(self.file_path)
            self.image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.image_width, self.image_height = img.size

    def zoom_in(self):
        self.xyxy = []
        self.xyxy4draw = []
        if self.image:
            self.zoom_factor *= 1.2
            self.update_image()

    def zoom_out(self):
        self.xyxy = []
        self.xyxy4draw = []
        if self.image:
            self.zoom_factor /= 1.2
            self.update_image()

    def update_image(self):
        if self.image:
            img = Image.open(self.file_path)
            width = int(img.width * self.zoom_factor)
            height = int(img.height * self.zoom_factor)
            img = img.resize((width, height), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(self.x_position, self.y_position, anchor=tk.NW, image=self.image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def move_x(self, event):
        if self.image:
            new_x_position = self.x_position + event.delta / 2
            max_scrollable_x = self.canvas.winfo_width() - self.image.width()
            if new_x_position <= 0 and new_x_position >= max_scrollable_x:
                self.x_position = new_x_position
                self.update_image()

    def move_y(self, event):
        if self.image:
            new_y_position = self.y_position + event.delta / 2
            max_scrollable_y = self.canvas.winfo_height() - self.image.height()
            if new_y_position <= 0 and new_y_position >= max_scrollable_y:
                self.y_position = new_y_position
                self.update_image()

    def on_mouse_move(self, event):
        self.canvas.delete("lines")
        x, y = event.x, event.y
        self.canvas.create_line(x, 0, x, self.canvas.winfo_height(), fill="red", tags="lines")
        self.canvas.create_line(0, y, self.canvas.winfo_width(), y, fill="red", tags="lines")

    def capture_click(self, event):
        if self.image:
            self.update_image()
            self.clicked_x_original = (event.x - self.x_position) / self.zoom_factor
            self.clicked_y_original = (event.y - self.y_position) / self.zoom_factor
            print(f"Clicked at ({self.clicked_x_original}, {self.clicked_y_original})")
            
            if self.left_buttons.selected_option.get()=="box":
                self.xyxy.append((self.clicked_x_original, self.clicked_y_original))
                self.xyxy4draw.append((event.x, event.y))
                if len(self.xyxy)==1:
                    self.add_dot_at_clicked_point()
                elif len(self.xyxy)==2:
                    self.draw_box()
                else:
                    # drop first 2 points
                    self.xyxy.pop(0)
                    self.xyxy.pop(0)
                    self.xyxy4draw.pop(0)
                    self.xyxy4draw.pop(0)
                    self.update_image()
                    self.add_dot_at_clicked_point()

            else:
                self.add_dot_at_clicked_point()


    def draw_box(self):
        self.canvas.create_rectangle(self.xyxy4draw)
    
    # ------ segment model functions ----
    def select_model(self):
        self.sam_checkpoint = filedialog.askopenfilename(filetypes=[("SAM Models", "*.pth")],
                                                    initialdir=".")
        if "vit_b" in self.sam_checkpoint:
            self.model_type = "vit_b"
        if "vit_h" in self.sam_checkpoint:
            self.model_type = "vit_h"
        
    def create_embeddings(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)

        if torch.cuda.is_available():
            print("CUDA is available:", torch.cuda.is_available())
            sam.to(device=device)
        
        img = cv2.imread(self.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(img)
        self.image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        messagebox.showinfo("Info", f"Embeddings Created - {self.image_embedding.shape}")

    def get_preds_point(self):
        x, y = self.clicked_x_original, self.clicked_y_original
        if x is None:
            messagebox.showerror("Error", f"Please select a point first")
            return

        input_label = np.array([1])
        input_point = np.array([[x, y]])
        self.masks, self.scores, self.logits = self.predictor.predict(
                                        point_coords=input_point,
                                        point_labels=input_label)
        

    def get_preds_box(self):
        if len(self.xyxy)!=2:
            messagebox.showerror("Error", f"Please draw a rectangle first")
            return
        input_label = np.array([1])
        input_box = np.array(self.xyxy)
        self.masks, self.scores, self.logits = self.predictor.predict(
                                box=input_box,
                                point_labels=input_label)
       

    def get_mask(self):
        if self.left_buttons.selected_option.get()=="point":
            self.get_preds_point()
        else:
            self.get_preds_box()

        img = cv2.imread(self.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        mask_index = np.argmax(self.scores)
        self.mask = self.masks[mask_index]
        self.mask = self.mask[np.newaxis,:]
        h, w = self.mask.shape[-2:]
        self.mask = self.mask.reshape(h, w)
        self.mask = self.mask.astype(np.uint8)
        
        self.show_masked_image()

    def show_masked_image(self):
        original_image = Image.open(self.file_path)
        colored_image = original_image.copy()
        # mask_array = np.load("vegetables_tomato.npy")
        mask_array = (self.mask > 0).astype(np.uint8) * 255
        color = (0, 255, 0)  # (R, G, B)
        mask_image = Image.fromarray(mask_array)
        
        colored_image.paste(Image.new("RGB", colored_image.size, color), mask=mask_image)
        
        width = int(colored_image.width * self.zoom_factor)
        height = int(colored_image.height * self.zoom_factor)
        img = colored_image.resize((width, height), Image.LANCZOS)
        
        self.image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.x_position, self.y_position, anchor=tk.NW, image=self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))





    def add_dot_at_clicked_point(self):
        if self.clicked_x_original is not None:
            x = self.clicked_x_original * self.zoom_factor + self.x_position
            y = self.clicked_y_original * self.zoom_factor + self.y_position
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")  # Create a red dot
        else:
            print("No clicked point available.")
            pass

    def create_mask_as_image(self):
        # to save object mask
        binary_mask_array = np.array(self.mask, dtype=np.uint8)  # Convert to 8-bit unsigned integer
        binary_mask_array[binary_mask_array == 1] = 255 # this can be skipped
        mask_2channel = binary_mask_array.reshape(self.image_height, self.image_width)
        return mask_2channel

    def save_mask(self):
        project_folder = os.path.join("Projects", 
                                      self.top_buttons.project_name_entry.get())
        curr_class_index = self.right_buttons.class_listbox.curselection()
        try:
            curr_class = self.class_names[curr_class_index[0]]
        except:
            messagebox.showerror("Error", f"Please select a class first")
            return

        class_folder = os.path.join(project_folder, curr_class)
        mask_ids = os.listdir(class_folder)
        if len(mask_ids)>0:
            mask_ids = [int(m.split("_")[0]) for m in mask_ids]
            mask_id = np.max(mask_ids) + 1
        else:
            mask_id = 0
        mask_name = f"{mask_id}_mask.json"
        mask_name_png = f"{mask_id}_mask.png"
        mask_path = os.path.join(class_folder, mask_name)
        mask_path_png = os.path.join(class_folder, mask_name_png)
        # mask_loaded = np.load('mask.npy')
        image_name_temp = os.path.basename(self.file_path)
        mask_dict = {"class" : curr_class, 
                    "image_name" : image_name_temp,
                    "point": (self.clicked_x_original,self.clicked_y_original),
                    "mask" : self.mask.tolist(),
                    }
        
        mask_2channel = self.create_mask_as_image()
        cv2.imwrite(mask_path_png, mask_2channel)
        
        with open(mask_path, 'w') as json_file:
            json.dump(mask_dict, json_file, indent=4)
        
        self.update_image()
        messagebox.showinfo("Info", f"Mask saved - Class: {curr_class}")


    # -------  right buttons ------
    def update_class_list(self):
        self.right_buttons.class_listbox.delete(0, tk.END)  # Clear the listbox
        for name in self.class_names:
            self.right_buttons.class_listbox.insert(tk.END, name)

    def add_class(self):
        class_name = self.right_buttons.class_name_entry.get()
        if class_name:
            project_folder = os.path.join("Projects", 
                                          self.top_buttons.project_name_entry.get())
            if not os.path.exists(project_folder):
                os.mkdir(project_folder) 

            class_folder = os.path.join(project_folder, class_name)
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)  # Create the parent folder if it doesn't exist        
            self.class_names.append(class_name)
            self.right_buttons.class_name_entry.delete(0, tk.END)  # Clear the entry field
            self.update_class_list()
              
    def remove_class(self):
        selected_index = self.right_buttons.class_listbox.curselection()
        if selected_index:
            project_folder = os.path.join("Projects", self.top_buttons.project_name_entry.get())
            curr_class_index = self.right_buttons.class_listbox.curselection()
            curr_class = self.class_names[curr_class_index[0]]
            class_folder = os.path.join(project_folder, curr_class)
            shutil.rmtree(class_folder)
            self.class_names.pop(selected_index[0])
            self.update_class_list()

    def load_old_class_list(self):
        project_folder = os.path.join("Projects", self.top_buttons.project_name_entry.get())
        self.class_names = os.listdir(project_folder)
        self.right_buttons.class_listbox.delete(0, tk.END)  # Clear the listbox
        for name in self.class_names:
            self.right_buttons.class_listbox.insert(tk.END, name)
    
    def read_saved_mask(self, mask_json_path):
        with open(mask_json_path, 'r') as json_file:
            mask_dict = json.load(json_file)
        return mask_dict

    
    def load_masked_points(self):
        original_image = Image.open(self.file_path)
        colored_image = original_image.copy()
    
        project_folder = os.path.join("Projects", self.top_buttons.project_name_entry.get())
        curr_class_index = self.right_buttons.class_listbox.curselection()
        try:
            curr_class = self.class_names[curr_class_index[0]]
        except:
            messagebox.showerror("Error", f"Please select a class first")
        class_folder = os.path.join(project_folder, curr_class)
        mask_list = os.listdir(class_folder)
        mask_list = [mm for mm in mask_list if ".json" in mm]
        # clicked_points_temp = []
        for mm in mask_list:
            mask_json_path = os.path.join(class_folder, mm)
            mask_dict = self.read_saved_mask(mask_json_path)
            mask_temp = np.array(mask_dict["mask"])
            mask_array = (mask_temp > 0).astype(np.uint8) * 255
            color = (0, 255, 0)  # (R, G, B)
            mask_image = Image.fromarray(mask_array)
            
            colored_image.paste(Image.new("RGB", colored_image.size, color), mask=mask_image)

        width = int(colored_image.width * self.zoom_factor)
        height = int(colored_image.height * self.zoom_factor)
        img = colored_image.resize((width, height), Image.LANCZOS)
        
        self.image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.x_position, self.y_position, anchor=tk.NW, image=self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
