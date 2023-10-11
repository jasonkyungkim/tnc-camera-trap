import os
import shutil

base_dir = "images/images/Set1"
unsorted_dir = os.path.join(base_dir, "processed2")
processed_dir = os.path.join(base_dir, "processed")

# Iterate through every file in the unsorted directory
for filename in os.listdir(unsorted_dir):
    if filename.endswith((".jpg", ".JPG", ".jpeg", ".png")):
        seq_prefix = filename.split("_")[0]  # Assuming filenames are in the format SEQXXXX_...

        # Search in Set1 for the appropriate ANIMAL/SEQXXXX folder
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if any(seq_prefix in d for d in dirnames):
                # Get the relative path to determine the ANIMAL name and reconstruct the folder in 'processed'
                rel_path = os.path.relpath(dirpath, base_dir)
                destination_dir = os.path.join(processed_dir, rel_path, seq_prefix)

                # Create the directory if it doesn't exist
                os.makedirs(destination_dir, exist_ok=True)

                # Move the image and corresponding txt file to the new directory
                shutil.move(os.path.join(unsorted_dir, filename), os.path.join(destination_dir, filename))
                shutil.move(os.path.join(unsorted_dir, os.path.splitext(filename)[0] + ".txt"), os.path.join(destination_dir, os.path.splitext(filename)[0] + ".txt"))
                break

print("Sorting complete!")
