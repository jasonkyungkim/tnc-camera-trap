import os

base_dir = "images/images/Set1"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Gather list of class names from the train directory (should be same for val directory too)
class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# Generate the yaml content
yaml_content = f"""train: {train_dir}/
val: {val_dir}/

nc: {len(class_names)}
names: {class_names}
"""

# Write to data.yaml
with open(os.path.join(base_dir, "data.yaml"), 'w') as f:
    f.write(yaml_content)

print("data.yaml generated!")
