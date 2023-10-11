import os
from PIL import Image
import shutil

base_dir = "images/images/Set1"
output_base_dir = os.path.join(base_dir, "processed")

# Function to resize images
def resize_image(input_path, output_path, size=(640, 640)):
    try:
        with Image.open(input_path) as img:
            img_resized = img.resize(size, Image.ANTIALIAS)
            img_resized.save(output_path)
            #print(f"Resized and saved: {output_path}")
    except OSError as e:
        print(f"Failed to process {input_path} due to: {e}")

# Traverse the directory tree
for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith((".jpg", ".JPG", ".jpeg", ".png")):
            image_path = os.path.join(dirpath, filename)
            txt_path = os.path.splitext(image_path)[0] + ".txt"

            # Generate the new path inside 'processed' while preserving the folder structure
            relpath = os.path.relpath(dirpath, base_dir)
            output_dir = os.path.join(output_base_dir, relpath)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            resized_image_path = os.path.join(output_dir, filename)
            new_txt_path = os.path.join(output_dir, os.path.basename(txt_path))

            # Check if the corresponding .txt file exists
            if os.path.exists(txt_path):
                #print(f"Found matching .txt for: {filename}")

                # Resize and save the image to the new directory
                resize_image(image_path, resized_image_path)

                # Move the .txt file to the new directory
                shutil.move(txt_path, new_txt_path)
                #print(f"Moved {txt_path} to {new_txt_path}")
            #else:
                #print(f"No matching .txt file for: {filename}")
                #print(f"Looked for: {txt_path}")

print("Processing complete!")


