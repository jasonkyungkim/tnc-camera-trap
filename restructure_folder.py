import os
import shutil

base_dir = "/Users/ll/Harvard/Capstone/tnc-camera-trap/images/images/Set1/"  # Replace with the path to your Set1 directory
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
