import os

def remove_empty_annotations(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file:
                    if not file.read().strip():  # Check if the file is empty
                        # Remove the .txt file
                        os.remove(filepath)
                        print(f"Removed empty label file: {filepath}")

                        # Remove the associated image
                        image_formats = ['.jpg', '.jpeg', '.png']  # You can expand this list if needed
                        for ext in image_formats:
                            image_path = os.path.join(root, filename.replace('.txt', ext))
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                print(f"Removed associated image: {image_path}")
                                break  # Only delete one corresponding image


base_dir = "/Users/ll/Harvard/Capstone/tnc-camera-trap/images/images/Set1/train"
remove_empty_annotations(base_dir)


