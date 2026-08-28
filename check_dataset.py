import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# ==========================================
# Image transformation
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==========================================
# Load training dataset
# ==========================================

dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)


# ==========================================
# Print dataset information
# ==========================================

print("Classes:")
print(dataset.classes)

print("\nClass mapping:")
print(dataset.class_to_idx)

print("\nNumber of training images:")
print(len(dataset))


# ==========================================
# Create image grid
# ==========================================

figure = plt.figure(figsize=(12, 8))


for i in range(8):

    image, label = dataset[i]

    image = image.permute(1, 2, 0)

    ax = figure.add_subplot(2, 4, i + 1)

    ax.imshow(image)

    ax.set_title(
        dataset.classes[label]
    )

    ax.axis("off")


plt.tight_layout()


# ==========================================
# Save preview
# ==========================================

plt.savefig(
    "dataset_preview.png",
    dpi=150
)

print("\nPreview saved as:")
print("dataset_preview.png")
