# MobileNetV2 Image Classification

This project uses **MobileNetV2** for image classification.

## 📋 Prerequisites

Make sure you have **Python 3** installed on your system.

## 🚀 Setup

### Windows

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

### Linux

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

## 📦 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🗂️ Prepare the Dataset

Run the following command to prepare the dataset:

```bash
python prepare_dataset.py
```

## 🔍 Check the Dataset

Verify that the dataset is prepared correctly:

```bash
python check_dataset.py
```

## 🏋️ Train the Model

Train the MobileNetV2 model:

```bash
python train.py
```

## 🔮 Make Predictions

To predict the class of an image, run:

```bash
python predict.py Images/img2.jpg
```

Replace `Images/img2.jpg` with the path to your own image.

## 📁 Project Workflow

The overall workflow is:

```text
1. Create and activate virtual environment
2. Install dependencies
3. Prepare dataset
4. Check dataset
5. Train MobileNetV2 model
6. Predict on new images
```

## 📝 Commands Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset
python prepare_dataset.py

# Check dataset
python check_dataset.py

# Train model
python train.py

# Predict
python predict.py Images/img2.jpg
```
