import tkinter as tk

class TopButtons:
    def __init__(self, frame, image_viewer):
        self.frame = frame
        self.image_viewer = image_viewer
        self.create_buttons()


    def create_buttons(self): 
        self.project_name_label = tk.Label(self.frame, text="Enter Project Name:")
        self.project_name_label.pack(side=tk.LEFT, padx=10)
        self.project_name_entry = tk.Entry(self.frame)
        self.project_name_entry.pack(side=tk.LEFT, padx=10)
        self.project_name_entry.insert(0, "Project1")
        
        self.load_button = tk.Button(self.frame, text="Load Image", command=self.image_viewer.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.zoom_in_button = tk.Button(self.frame, text="Zoom In", command=self.image_viewer.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=10)
        self.zoom_out_button = tk.Button(self.frame, text="Zoom Out", 
                                         command=self.image_viewer.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=10)


class BottomButtons:
    def __init__(self, frame, image_viewer):
        self.frame = frame
        self.image_viewer = image_viewer
        self.create_buttons()


    def create_buttons(self):         
        self.reset_button = tk.Button(self.frame, text="Reset", 
                                      command=self.image_viewer.reset_variables)
        self.reset_button.pack(side=tk.TOP, pady=5)      
        
        self.note_label = tk.Label(self.frame, 
                                   text="Use the mouse wheel to move along the X and Shift + mouse wheel for the Y axes.")
        self.note_label.pack(side=tk.TOP, pady=5)

class LeftButtons:
    def __init__(self, frame, image_viewer):
        self.frame = frame
        self.image_viewer = image_viewer
        
        self.create_buttons()

    def on_radio_select(self):
        self.selected_option.get()
        print("input_type :", self.selected_option.get())
        

    def create_buttons(self):
        self.input_type_label = tk.Label(self.frame, text="Select Input Type")
        self.input_type_label.pack(side=tk.TOP, padx=10)

        self.selected_option = tk.StringVar()
        # Create radio buttons
        self.point_radio = tk.Radiobutton(self.frame, text="Point", 
                                    variable=self.selected_option, value="point", command=self.on_radio_select)
        self.point_radio.pack()

        self.box_radio = tk.Radiobutton(self.frame, text="Box", 
                                variable=self.selected_option, value="box", command=self.on_radio_select)
        self.box_radio.pack()
        self.selected_option.set("point")


        self.select_model_button = tk.Button(self.frame, text="Select Model", 
                                             command=self.image_viewer.select_model)
        self.select_model_button.pack(side=tk.TOP, pady=10)

        
        self.create_embeddings_button = tk.Button(self.frame, text="Create Embeddings",
                                                  command=self.image_viewer.create_embeddings)
        self.create_embeddings_button.pack(side=tk.TOP, pady=10)

        self.get_mask_button = tk.Button(self.frame, text="Get Mask", 
                                             command=self.image_viewer.get_mask,
                                            #  command=self.image_viewer.get_preds,
                                             )
        self.get_mask_button.pack(side=tk.TOP, pady=10)


        self.save_mask_button = tk.Button(self.frame, text="Save Mask",
                                                  command=self.image_viewer.save_mask)
        self.save_mask_button.pack(side=tk.TOP, pady=10)

class RightButtons:
    def __init__(self, frame, image_viewer):
        self.frame = frame
        self.image_viewer = image_viewer
        self.create_buttons()

    def create_buttons(self):
        self.class_name_label = tk.Label(self.frame, text="Enter Class Name:")
        self.class_name_label.pack(side=tk.TOP, pady=10)

        self.class_name_entry = tk.Entry(self.frame)
        self.class_name_entry.pack(side=tk.TOP, pady=10)
        
        self.add_class_button = tk.Button(self.frame, text="Add Class", 
                                          command=self.image_viewer.add_class)
        self.add_class_button.pack(side=tk.TOP, pady=10)

        self.remove_class_button = tk.Button(self.frame, text="Remove Class", 
                                          command=self.image_viewer.remove_class)
        self.remove_class_button.pack(side=tk.TOP, pady=10)

        self.class_listbox = tk.Listbox(self.frame)
        self.class_listbox.pack(side=tk.TOP, pady=10)

        self.load_old_folders_button = tk.Button(self.frame, text="Load Existing Classes", 
                                          command=self.image_viewer.load_old_class_list)
        self.load_old_folders_button.pack(side=tk.TOP, pady=10)

        self.load_saved_masks_button = tk.Button(self.frame, text="Load Saved Masks", 
                                          command=self.image_viewer.load_masked_points)
        self.load_saved_masks_button.pack(side=tk.TOP, pady=10)

