import os
import shutil
import random
import kagglehub


# ==========================================
# Configuration
# ==========================================

TRAIN_IMAGES_PER_CLASS = 80
VAL_IMAGES_PER_CLASS = 20

random.seed(42)


# ==========================================
# Download / locate PlantVillage dataset
# ==========================================

print("Getting PlantVillage dataset...")

dataset_path = kagglehub.dataset_download(
    "emmarex/plantdisease"
)

print("\nDataset location:")
print(dataset_path)


# ==========================================
# Actual class names in this dataset
# ==========================================

classes = {
    "Tomato_healthy": "healthy",
    "Tomato_Early_blight": "early_blight"
}


# ==========================================
# Find the class directories
# ==========================================

class_directories = {}

for root, dirs, files in os.walk(dataset_path):

    for class_name in classes:

        if class_name in dirs:

            class_directories[class_name] = os.path.join(
                root,
                class_name
            )


# Check that both classes exist

for class_name in classes:

    if class_name not in class_directories:

        raise Exception(
            f"Could not find class: {class_name}"
        )


print("\nFound classes:")

for class_name, path in class_directories.items():

    print(f"{class_name} -> {path}")


# ==========================================
# Create output directories
# ==========================================

train_directory = os.path.join(
    "dataset",
    "train"
)

val_directory = os.path.join(
    "dataset",
    "val"
)


for output_class in classes.values():

    os.makedirs(
        os.path.join(
            train_directory,
            output_class
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            val_directory,
            output_class
        ),
        exist_ok=True
    )


# ==========================================
# Process each class
# ==========================================

for source_class, output_class in classes.items():

    source_directory = class_directories[
        source_class
    ]

    print(
        f"\nProcessing {source_class}..."
    )

    # Get images

    images = [
        file
        for file in os.listdir(source_directory)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"Available images: {len(images)}"
    )


    # Shuffle

    random.shuffle(images)


    required_images = (
        TRAIN_IMAGES_PER_CLASS
        + VAL_IMAGES_PER_CLASS
    )


    if len(images) < required_images:

        raise Exception(
            f"Not enough images for {source_class}"
        )


    # Select 100 images

    selected_images = images[
        :required_images
    ]


    # First 80 -> training

    train_images = selected_images[
        :TRAIN_IMAGES_PER_CLASS
    ]


    # Next 20 -> validation

    val_images = selected_images[
        TRAIN_IMAGES_PER_CLASS:
    ]


    # ======================================
    # Copy training images
    # ======================================

    for image in train_images:

        source = os.path.join(
            source_directory,
            image
        )

        destination = os.path.join(
            train_directory,
            output_class,
            image
        )

        shutil.copy2(
            source,
            destination
        )


    # ======================================
    # Copy validation images
    # ======================================

    for image in val_images:

        source = os.path.join(
            source_directory,
            image
        )

        destination = os.path.join(
            val_directory,
            output_class,
            image
        )

        shutil.copy2(
            source,
            destination
        )


    print(
        f"Created {output_class}: "
        f"{len(train_images)} train + "
        f"{len(val_images)} validation"
    )


# ==========================================
# Final result
# ==========================================

print("\n======================================")
print("Dataset preparation completed!")
print("======================================")

print("""
dataset/
│
├── train/
│   ├── healthy/
│   └── early_blight/
│
└── val/
    ├── healthy/
    └── early_blight/
""")

print("Training images:   160")
print("Validation images:  40")
print("Total images:      200")
