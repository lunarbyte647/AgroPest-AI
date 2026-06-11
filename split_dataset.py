import os
import shutil
import random

source = "dataset"

train_dir = "dataset_split/train"
val_dir = "dataset_split/val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in os.listdir(source):
    cls_path = os.path.join(source, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_imgs = images[:split]
    val_imgs = images[split:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img)
        )

print("Dataset Split Complete!")