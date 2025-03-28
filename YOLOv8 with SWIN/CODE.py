import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import timm
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset preparation
base_path = os.path.dirname(__file__)
images_path = os.path.join(base_path, "images") # Path to the folder containing images
labels_path = os.path.join(base_path, "labels") # Path to the folder containing labels
output_path = os.path.join(base_path, "yolo_dataset") # Path to the folder containing yolo formatted dataset

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)

# Parameters for data splitting
test_size = 0.2
val_size = 0.2

# Get list of images and corresponding labels
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

# Check if all images have corresponding labels
image_names = [os.path.splitext(f)[0] for f in image_files]
label_names = [os.path.splitext(f)[0] for f in label_files]

# Identify images missing corresponding labels
missing_labels = set(image_names) - set(label_names)
if missing_labels:
    print(f"Warning: Missing labels for images: {missing_labels}")
     # Exclude images without labels
    image_files = [f for f in image_files if os.path.splitext(f)[0] not in missing_labels]

# Splitting the dataset
train_imgs, test_imgs = train_test_split(image_files, test_size=test_size, random_state=42)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)

# Function to copy files
def copy_files(image_list, split):
    for img_file in image_list:
        # Copy image
        shutil.copy(os.path.join(images_path, img_file),
                    os.path.join(output_path, 'images', split, img_file))
        # Copy corresponding label
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.copy(os.path.join(labels_path, label_file),
                        os.path.join(output_path, 'labels', split, label_file))

# Copy the train, val, and test folders with respective data
copy_files(train_imgs, 'train')
copy_files(val_imgs, 'val')
copy_files(test_imgs, 'test')

# Defining YAML 
import yaml
data_yaml = {
    'train': os.path.relpath(os.path.join(output_path, 'images/train'), base_path),
    'val': os.path.relpath(os.path.join(output_path, 'images/val'), base_path),
    'test': os.path.relpath(os.path.join(output_path, 'images/test'), base_path),
    'nc': 8,
    'names': [
        'Quartzity', 'Live_Knot', 'Marrow', 'Resin',
        'Dead_Knot', 'Knot_with_Crack', 'Knot_Missing', 'Crack'
    ]
}

# Save the YAML configuration file
yaml_file_path = os.path.join(base_path, 'data.yaml')
with open(yaml_file_path, 'w') as file:
    yaml.dump(data_yaml, file)
print(f"data.yaml saved at: {yaml_file_path}")

# Data Augmentation using Albumentations
augmentation_pipeline = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.Resize(640, 640),
    ToTensorV2()
])

# Custom YOLOv8 model with Swin Transformer backbone
class CustomYOLOv8(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Replace backbone with Swin Transformer
        self.model.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        # Adjust output to match YOLO's output
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'out_channels'):
            self.model.backbone.num_classes = self.model.head.out_channels

# Initialize model
model = CustomYOLOv8('yolov8n.pt')  # Start with YOLOv8 checkpoint

# Train the model
model.train(
    data=yaml_file_path,
    epochs=100,                    
    imgsz=640,
    batch=16,
    name='yolov8_swin_transformer',
    workers=8
)

# Evaluate the model
metrics = model.val()
print(metrics)

# Run inference on test data
results = model.predict(
    source=os.path.join(output_path, 'images/test'),
    save=True,
    imgsz=640,
    augment=True
)

# Print predictions
for result in results:
    print(result)