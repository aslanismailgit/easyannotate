
#%%
import cv2
import numpy as np
import json
import os
import shutil
import math

class YoloTextOut:
    def __init__(self):
        self.yolo_images_path = None
        self.yolo_labels_path = None
        self.w = None
        self.h = None


    def read_saved_mask(self, mask_json_path):
        with open(mask_json_path, 'r') as json_file:
            mask_dict = json.load(json_file)
        return mask_dict

    def prep_yolo_out_dir(self, project_name, 
                          project_folder, valid_extensions):
        
        self.yolo_images_path = f"yolo/{project_name}/images/"
        self.yolo_labels_path = f"yolo/{project_name}/labels/"
        self.images_list = os.listdir(project_folder)
        self.images_list = [f for f in self.images_list if f.endswith(tuple(valid_extensions))]

        if not os.path.exists(self.yolo_images_path):
            os.makedirs(self.yolo_images_path)
        if not os.path.exists(self.yolo_labels_path):
            os.makedirs(self.yolo_labels_path)
        return True

    def create_classes_text(self, project_folder, class_names):
        # class names order should be the same as class ids at txt files
        class_text_path = project_folder + 'classes.txt'
        # print(class_text_path)
        with open(class_text_path, 'w') as file:
            for element in class_names:
                file.write(element + '\n')
        return True

    def get_polygon_points(self, class_id, mask):
        mask = np.array(mask, np.uint8)
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        np.unique(binary_mask)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        len(contours)

        # Assuming there is only one contour, you can access it
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to get a polygon
            epsilon = 0.0051 * cv2.arcLength(largest_contour, True)
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            divided_points = [[point[0] / self.w, point[1] / self.h] for point in np.squeeze(approx_polygon)]

            # sort points clockwise
            # Calculate the centroid of the points
            centroid_x = sum(x for x, _ in divided_points) / len(divided_points)
            centroid_y = sum(y for _, y in divided_points) / len(divided_points)

            # Define a function to calculate the polar angle of a point with respect to the centroid
            def polar_angle(point):
                x, y = point
                dx = x - centroid_x
                dy = y - centroid_y
                return math.atan2(dy, dx)

            # Sort the points by their polar angle in a clockwise order
            sorted_points = sorted(divided_points, key=polar_angle)
            sorted_points = np.array(sorted_points).flatten()
            yolo_txt_line = ' '.join(map(str, sorted_points))
            yolo_txt_line = str(class_id) + " " + yolo_txt_line + "\n"
            return yolo_txt_line
        

    def save_as_yolo_text(self, project_folder, class_names):
        for i in range(len(self.images_list)):
            # bu images üzerinde loop
            image_temp = self.images_list[i]
            # print("--- i ", i)
            # print(image_temp)
            # copy image_temp to yolo_images_path
            src_image = os.path.join(project_folder, image_temp)
            dst_image = os.path.join(self.yolo_images_path, image_temp)
            shutil.copy(src_image, dst_image)
            image_org = cv2.imread(src_image)
            self.h, self.w, _ = image_org.shape
            image_basename = image_temp.split(".")[0]
        
            for j in range(len(class_names)):
                # print("--------- j ", j)
                class_curr = class_names[j]
                class_curr_path = os.path.join(project_folder, class_curr)
                
                jsons_inthis_class = os.listdir(class_curr_path)
                jsons_inthis_class = [f for f in jsons_inthis_class if image_basename in f]
                jsons_inthis_class = [f for f in jsons_inthis_class if ".json" in f]
                jsons_inthis_class


                yolo_txt_path = os.path.join(self.yolo_labels_path, image_basename + ".txt")
                for k in range(len(jsons_inthis_class)):
                    # print("-------------- k ", k)
                    txt_mode = "a" if os.path.exists(yolo_txt_path) else "w"
                    json_temp = jsons_inthis_class[k]
                    json_temp_path = os.path.join(class_curr_path, json_temp)
                    # print(json_temp_path, "--------")
                    mask_json = self.read_saved_mask(json_temp_path)
                    mask = mask_json["mask"]
                    class_id = class_names.index(mask_json["class"])
                    # print(class_id, mask_json["class"])
                    yolo_txt_line = self.get_polygon_points(class_id, mask)
                    # print(yolo_txt_line)
                    # Open the file in write mode ('w')
                    with open(yolo_txt_path, txt_mode) as file:
                        file.write(yolo_txt_line)