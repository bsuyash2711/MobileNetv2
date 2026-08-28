import sys
import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# Classes
classes = [
    "early_blight",
    "healthy"
]


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Create MobileNetV2
model = models.mobilenet_v2(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)


# Load our trained model
model.load_state_dict(
    torch.load(
        "mobilenetv2_disease.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Get image path from command line
if len(sys.argv) < 2:
    print("Usage:")
    print("python predict.py <image_path>")
    sys.exit()


image_path = sys.argv[1]


# Check image
if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit()


# Load image
image = Image.open(image_path).convert("RGB")


# Preprocess
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)


# Prediction
with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )


# Result
predicted_class = classes[predicted.item()]
confidence = confidence.item() * 100


print("\n==============================")
print("       PREDICTION")
print("==============================")

print("Image:", image_path)
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("==============================")
