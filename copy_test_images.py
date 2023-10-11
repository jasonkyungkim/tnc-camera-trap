import os
import shutil
import random
import csv
from PIL import Image

base_dir = "images/images/Set1"
output_dir = "sampled_images"
csv_filename = "animal_image_mapping.csv"
num_samples_per_species = 4  # This number will change based on the total number of species and desired samples

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# List animal species directories
animal_species = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Sample images and copy to new directory, also collect data for csv
csv_data = []
for species in animal_species:
    species_dir = os.path.join(base_dir, species)
    seq_dirs = [os.path.join(species_dir, d) for d in os.listdir(species_dir) if os.path.isdir(os.path.join(species_dir, d))]
    all_images = [os.path.join(seq_dir, img) for seq_dir in seq_dirs for img in os.listdir(seq_dir) if img.endswith('.JPG')]

    sampled_images = random.sample(all_images, min(num_samples_per_species, len(all_images)))
    for img_path in sampled_images:
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_dir, img_name))

        # Resize the image if desired (this step can be skipped if not necessary)
        # img = Image.open(os.path.join(output_dir, img_name))
        # img = img.resize((640, 640))  # Resize to YOLO's format, change dimensions if needed
        # img.save(os.path.join(output_dir, img_name))

        csv_data.append([species, img_name])

# Save csv data
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ANIMAL NAME", "IMAGE ID"])  # Writing headers
    writer.writerows(csv_data)

print(f"Images copied to {output_dir} and CSV saved as {csv_filename}")
