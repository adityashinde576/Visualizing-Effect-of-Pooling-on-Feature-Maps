# CNN Feature Maps Visualization (PyTorch)

## Project Overview

This project demonstrates how Convolutional Neural Networks (CNNs) extract features from images and how pooling operations affect those features. Using a single image from the CIFAR-10 dataset, the project visualizes:

* Feature maps **before pooling**
* Feature maps **after max pooling**
* Feature maps **after average pooling**

The goal is educational: to clearly understand the role of convolution, ReLU activation, and pooling layers in CNNs.

---

## Project Structure

```
.
├── notebook.ipynb        # Jupyter Notebook containing the full implementation
├── data/                # CIFAR-10 dataset (auto-downloaded)
└── README.md             # Project documentation
```

---

## Technologies Used

* Python 3.8+
* PyTorch
* Torchvision
* Matplotlib
* Jupyter Notebook / JupyterLab

---

## Concepts Covered

* Convolutional layers (`nn.Conv2d`)
* ReLU activation function
* Max Pooling (`nn.MaxPool2d`)
* Average Pooling (`nn.AvgPool2d`)
* Feature map visualization
* CIFAR-10 dataset handling

---

## Installation Instructions

### 1. Create and Activate Conda Environment (Recommended)

```
conda create -n cnn_feature_maps python=3.9 -y
conda activate cnn_feature_maps
```

### 2. Install Required Libraries

```
pip install torch torchvision matplotlib jupyter
```

Verify installation:

```
python -c "import torch; print(torch.__version__)"
```

---

## How to Run the Project

### Step 1: Launch Jupyter Notebook

```
jupyter notebook
```

or

```
jupyter lab
```

### Step 2: Open the Notebook

Open the provided notebook file and run the cells sequentially from top to bottom.

---

## Code Workflow Explanation

### 1. Import Libraries

The project imports PyTorch, Torchvision, Matplotlib, and required modules.

### 2. Data Loading

* CIFAR-10 dataset is downloaded automatically
* Only **one test image** is used for visualization
* Image is converted to a tensor

### 3. Model Components

The project uses individual layers instead of a full CNN model:

* `Conv2d`: Extracts spatial features
* `ReLU`: Introduces non-linearity
* `MaxPool2d`: Captures strongest activations
* `AvgPool2d`: Smooths spatial information

### 4. Forward Pass

```
feature_before_pool = relu(conv(image))
feature_max_pool = max_pool(feature_before_pool)
feature_avg_pool = avg_pool(feature_before_pool)
```

### 5. Feature Map Visualization

A custom function plots feature maps using Matplotlib:

* Batch dimension is removed using `squeeze(0)`
* Multiple channels are displayed side-by-side

---

## Output Description

### Before Pooling

* High-resolution feature maps
* Retains detailed spatial information

### After Max Pooling

* Reduced spatial dimensions
* Strongest activations preserved

### After Average Pooling

* Reduced spatial dimensions
* Smoother and more generalized features

---

## Expected Output

* Three rows of grayscale images:

  * Before Pooling
  * After Max Pooling
  * After Average Pooling
* Each row displays multiple feature maps from different channels

---

## Common Issues and Fixes

### Dataset Download Error

Ensure internet access is available and `download=True` is set.

### Shape Errors

Make sure:

```
image = image.unsqueeze(0)
```

CNN layers expect input shape:

```
[batch_size, channels, height, width]
```

### Blank Feature Maps

Some filters may not activate strongly. This is expected behavior.

---

## Learning Outcomes

After completing this project, you will understand:

* How CNNs extract features from images
* Difference between max pooling and average pooling
* How pooling impacts spatial resolution
* How to visualize intermediate CNN layers

---

## Future Enhancements

* Add multiple convolution layers
* Visualize feature maps across deeper layers
* Compare different kernel sizes
* Apply normalization and color maps

---

## Author

This project is intended for learning and demonstration purposes in Deep Learning and Computer Vision using PyTorch.

---

## License

This project is free to use for educational purposes.
