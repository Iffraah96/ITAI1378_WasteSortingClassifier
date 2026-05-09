import zipfile

zip_path = "/content/archive.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

print("Dataset extracted successfully!")

import os

os.listdir("/content/dataset")

import os
from pathlib import Path

# Common paths:
# Local/Colab after unzip:
#DATASET_PATH = Path("garbage-classification/Garbage classification/Garbage classification")
DATASET_PATH = Path("/content/dataset/garbage classification/Garbage classification")

# Kaggle notebook example:
# DATASET_PATH = Path("/kaggle/input/garbage-classification/Garbage classification/Garbage classification")

print("Dataset path exists:", DATASET_PATH.exists())
print("Dataset path:", DATASET_PATH)

if DATASET_PATH.exists():
    print("Folders:", sorted([p.name for p in DATASET_PATH.iterdir() if p.is_dir()]))
else:
    print("Please update DATASET_PATH to the folder that contains the class subfolders.")

class_counts = {}

if DATASET_PATH.exists():
    for class_folder in sorted(DATASET_PATH.iterdir()):
        if class_folder.is_dir():
            image_files = list(class_folder.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            class_counts[class_folder.name] = len(image_files)

    counts_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image Count"])
    display(counts_df)

    plt.figure(figsize=(8, 5))
    plt.bar(counts_df["Class"], counts_df["Image Count"])
    plt.title("Number of Images per Class")
    plt.xlabel("Waste Category")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Dataset path not found. Please fix DATASET_PATH first.")

if DATASET_PATH.exists():
    plt.figure(figsize=(12, 8))
    classes = sorted([p.name for p in DATASET_PATH.iterdir() if p.is_dir()])

    image_index = 1
    for class_name in classes:
        class_folder = DATASET_PATH / class_name
        image_files = [f for f in class_folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(image_files) > 0:
            sample_image = random.choice(image_files)
            img = tf.keras.utils.load_img(sample_image, target_size=(224, 224))

            plt.subplot(2, 3, image_index)
            plt.imshow(img)
            plt.title(class_name)
            plt.axis("off")
            image_index += 1

    plt.tight_layout()
    plt.show()
else:
    print("Dataset path not found.")
