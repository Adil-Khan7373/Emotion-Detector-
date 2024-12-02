import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = 'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\train'  # Path to your original train data
dest_train_dir = 'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\train 2'  # Path for new train data
dest_val_dir = 'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\datasets\val'  # Path for validation data

# Create directories for the new train and validation sets
os.makedirs(dest_train_dir, exist_ok=True)
os.makedirs(dest_val_dir, exist_ok=True)

# Get list of classes (emotion categories)
classes = os.listdir(source_dir)

# Split each class into train and validation
for class_name in classes:
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path):  # Ensure the path is a directory
        all_images = os.listdir(class_path)

        # Split the list into train and validation (80% train, 20% validation)
        train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

        # Move images to respective directories
        os.makedirs(os.path.join(dest_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(dest_val_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(dest_train_dir, class_name, img))

        for img in val_images:
            shutil.move(os.path.join(class_path, img), os.path.join(dest_val_dir, class_name, img))

print("Data split into train and validation sets successfully!")
