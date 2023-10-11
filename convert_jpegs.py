from PIL import Image
import os
import re

base_dir = "tnc-camera-trap/images/images/Set1"
labels_path = os.path.join(base_dir, "labels.txt")

# Generate class_map dictionary
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
class_map = {subdir: idx for idx, subdir in enumerate(subdirs)}
print(class_map)

with open(labels_path, 'r') as f:
    for line in f.readlines():
        elements = line.strip().split()

        img_path = os.path.join(base_dir, elements[0])
        
        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        # Determine Class ID from folder name
        animal_name = elements[0].split('/')[0]  # Extract the first part of the path
        class_id = class_map.get(animal_name, -1)  # Get the class ID

        #Load images and get dimensions

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        num_boxes = int(elements[1])
        yolo_annotations = []

        #Convert bounding boxes to YOLO format

        for i in range(num_boxes):
            xmin = float(elements[2 + i*4])
            ymin = float(elements[3 + i*4])
            xmax = float(elements[4 + i*4])
            ymax = float(elements[5 + i*4])

            x_center = (xmin + xmax) / 2.0 /img_width
            y_center = (ymin + ymax) / 2.0 /img_height 
            width = (xmax -xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        #Save to .txt file
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        print(f"Saving to: {txt_path}")
        with open(txt_path, 'w') as txt_f:
            txt_f.write("\n".join(yolo_annotations))
            