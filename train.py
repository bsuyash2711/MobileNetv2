import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Image transformations
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. Load datasets
# ==========================================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)


print("\nClasses:")
print(train_dataset.classes)

print("\nTraining images:")
print(len(train_dataset))

print("\nValidation images:")
print(len(val_dataset))


# ==========================================
# 4. DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)


val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)


# ==========================================
# 5. Load MobileNetV2
# ==========================================

print("\nLoading MobileNetV2...")


model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)


# ==========================================
# 6. Replace classifier
# ==========================================

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)


model = model.to(device)


# ==========================================
# 7. Loss function
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 8. Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# 9. Training
# ==========================================

num_epochs = 5


for epoch in range(num_epochs):

    print(
        f"\nEpoch {epoch + 1}/{num_epochs}"
    )

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    train_loss = (
        running_loss /
        len(train_loader)
    )


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    val_loss_total = 0.0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    val_loss = (
        val_loss_total /
        len(val_loader)
    )


    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# ==========================================
# 10. Save model
# ==========================================

torch.save(
    model.state_dict(),
    "mobilenetv2_disease.pth"
)


print("\nTraining completed!")

print(
    "Model saved as: "
    "mobilenetv2_disease.pth"
)
