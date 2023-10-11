import os
import random
import shutil

def is_annotation_empty(filepath):
    with open(filepath, 'r') as file:
        return not file.read().strip()

base_dir = "images/images/Set1/processed"
output_dir = "images/images/Set1"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

split_ratio = 0.8
animal_classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for animal_class in animal_classes:
    animal_class_dir = os.path.join(base_dir, animal_class)
    seq_folders = [seq for seq in os.listdir(animal_class_dir) if os.path.isdir(os.path.join(animal_class_dir, seq))]

    for seq_folder in seq_folders:
        seq_dir = os.path.join(animal_class_dir, seq_folder)
        images = [f for f in os.listdir(seq_dir) if f.endswith('.JPG')]

        valid_images = [img for img in images if not is_annotation_empty(os.path.join(seq_dir, os.path.splitext(img)[0] + '.txt'))]

        random.shuffle(valid_images)
        train_size = int(len(valid_images) * split_ratio)

        train_images = valid_images[:train_size]
        val_images = valid_images[train_size:]

        # Make sure class and seq subdirectories exist in train and val directories
        os.makedirs(os.path.join(train_dir, animal_class, seq_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, animal_class, seq_folder), exist_ok=True)

        # Copy images into respective split directories
        for image in train_images:
            shutil.copy(os.path.join(seq_dir, image), os.path.join(train_dir, animal_class, seq_folder, image))
            shutil.copy(os.path.join(seq_dir, os.path.splitext(image)[0] + '.txt'), os.path.join(train_dir, animal_class, seq_folder, os.path.splitext(image)[0] + '.txt'))

        for image in val_images:
            shutil.copy(os.path.join(seq_dir, image), os.path.join(val_dir, animal_class, seq_folder, image))
            shutil.copy(os.path.join(seq_dir, os.path.splitext(image)[0] + '.txt'), os.path.join(val_dir, animal_class, seq_folder, os.path.splitext(image)[0] + '.txt'))

# --- YOLO Transformation (from Script 3) ---

base_dir = "/Users/ll/Harvard/Capstone/tnc-camera-trap/images/images/Set1/"
splits = ['train', 'val']

for split in splits:
    split_dir = os.path.join(base_dir, split)
    
    # Creating the new structure directories if they don't exist
    new_images_dir = os.path.join(split_dir, 'images')
    new_labels_dir = os.path.join(split_dir, 'labels')
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)

    # List of all animal name directories in the current split
    animal_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)) and d not in ['images', 'labels']]

    for animal in animal_dirs:
        animal_path = os.path.join(split_dir, animal)
        seq_dirs = [d for d in os.listdir(animal_path) if os.path.isdir(os.path.join(animal_path, d))]

        for seq in seq_dirs:
            seq_path = os.path.join(animal_path, seq)
            files = os.listdir(seq_path)

            for file in files:
                src_path = os.path.join(seq_path, file)

                if file.endswith(('.JPG')):
                    dst_path = os.path.join(new_images_dir, file)
                    shutil.move(src_path, dst_path)
                elif file.endswith('.txt'):
                    dst_path = os.path.join(new_labels_dir, file)
                    shutil.move(src_path, dst_path)

    # After processing all SEQ directories for an animal, you can remove the animal directory since it will be empty
    for animal in animal_dirs:
        shutil.rmtree(os.path.join(split_dir, animal))


print("Processing complete!")
