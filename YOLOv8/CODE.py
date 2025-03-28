import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_path = os.path.dirname(os.path.abspath(__file__))   # Path to your dataset folder
images_path = os.path.join(base_path, "images")  # Path to images folder
labels_path = os.path.join(base_path, "labels")  # Path to labels folder
output_path = os.path.join(base_path, "yolo_dataset")  # Path to the folder for the YOLO-formatted dataset

# Create output directories for YOLO dataset
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)  # directory for images
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)  # directory for labels

# Parameters for data splitting
test_size = 0.2  # Proportion of dataset for test
val_size = 0.2   # Proportion of remaining data for validation

# Get list of images and corresponding labels
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

# Check if all images have corresponding labels
image_names = [os.path.splitext(f)[0] for f in image_files]
label_names = [os.path.splitext(f)[0] for f in label_files]

# Identify images without corresponding labels
missing_labels = set(image_names) - set(label_names)
if missing_labels:
    print(f"Warning: Missing labels for images: {missing_labels}")
    # Exclude images with out corresponding labels
    image_files = [f for f in image_files if os.path.splitext(f)[0] not in missing_labels] 

# Splitting the dataset
train_imgs, test_imgs = train_test_split(image_files, test_size=test_size, random_state=42)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)

# Helper function to copy files
def copy_files(image_list, split):
    for img_file in image_list:
        # Copy image to split folders
        shutil.copy(os.path.join(images_path, img_file),
                    os.path.join(output_path, 'images', split, img_file))
        # identify and Copy corresponding label
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.copy(os.path.join(labels_path, label_file),
                        os.path.join(output_path, 'labels', split, label_file))

copy_files(train_imgs, 'train') # Copy training data
copy_files(val_imgs, 'val')  # Copy validation data
copy_files(test_imgs, 'test') # Copy test data


import yaml

# Define the data for the YAML file
data_yaml = {
    'train': os.path.join('yolo_dataset', 'images', 'train'), # Path to training images
    'val': os.path.join('yolo_dataset', 'images', 'val'), # Path to validation images
    'test': os.path.join('yolo_dataset', 'images', 'test'),   # Path to test images
    'nc': 8,  # Number of classes
    'names': [    # Class names
        'Quartzity',
        'Live_Knot',
        'Marrow',
        'Resin',
        'Dead_Knot',
        'Knot_with_Crack',
        'Knot_Missing',
        'Crack'
    ]
}

# Save to a YAML file
yaml_file_path = os.path.join(base_path, 'data.yaml')
with open(yaml_file_path, 'w') as file:
    yaml.dump(data_yaml, file)

print(f"data.yaml saved at: {yaml_file_path}")

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n', 'yolov8s', 'yolov8m', etc., based on your system resources

# Train the model
model.train(
    data=r'data.yaml',  # Path to your data.yaml file
    epochs=50,                             # Number of training epochs
    imgsz=640,                             # Image size for training
    batch=16,                              # Batch size
    name='wood_defect_detection',           # Experiment name
    workers=8                              # Number of workers for data loading
)

# Evaluate the model
metrics = model.val()
print(metrics)  # Outputs mAP, precision, recall, etc.

# Load the trained model
model = YOLO(r'runs/detect/wood_defect_detection/weights/best.pt')  # Path to best weights

# Run inference
results = model.predict(
    source=os.path.join('yolo_dataset', 'images', 'test'),  # path to test data
    save=True,                                  # Save predictions
    imgsz=640,                                   # Image size for inference
)

# Print predictions
for result in results:
    print(result)
