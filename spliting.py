import os
import shutil
import random

source_dir = "dataset"
train_dir = "dataset/train"
val_dir = "dataset/val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 80% training, 20% validation
split_ratio = 0.8

for category in ["Bacterial leaf blight", "Brown spot", "Leaf smut"]:
    src_path = os.path.join(source_dir, category)
    train_path = os.path.join(train_dir, category)
    val_path = os.path.join(val_dir, category)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    files = os.listdir(src_path)
    random.shuffle(files)
    split_point = int(len(files) * split_ratio)

    train_files = files[:split_point]
    val_files = files[split_point:]

    for f in train_files:
        shutil.copy(os.path.join(src_path, f), os.path.join(train_path, f))
    for f in val_files:
        shutil.copy(os.path.join(src_path, f), os.path.join(val_path, f))

print("✅ Dataset split complete!")
