
# Retina Vessel Segmentation using U-Net

This project focuses on the segmentation of retinal blood vessels from grayscale fundus images using a U-Net-based convolutional neural network. Retinal vessel segmentation plays a key role in diagnosing diabetic retinopathy, glaucoma, and cardiovascular diseases. By training a semantic segmentation model, this project demonstrates how deep learning can extract meaningful vascular features in an end-to-end automated workflow.


## ⚙️ Tech Stack
- Python 3.x

- TensorFlow / Keras

- NumPy, OpenCV, Matplotlib

- scikit-learn

- Colab Notebook




## 📁 Dataset
📦 Dataset Used: Retina Blood Vessel Dataset by Abdallah Wagih

👁 Contents:

Grayscale retinal images

Corresponding binary masks for blood vessels

Train/Test Split provided


## 🧠 Model Architecture
The model is a U-Net-inspired encoder–decoder convolutional neural network, custom-built to perform pixel-wise segmentation of retinal blood vessels from grayscale fundus images.

U-Net is a powerful architecture originally developed for biomedical image segmentation and is known for its ability to work well with limited data while maintaining high spatial resolution through skip connections.

🔽 Encoder (Contracting Path):
The encoder progressively captures semantic information from the input image by applying convolution and pooling operations. It compresses the image spatially while expanding the number of filters to capture complex patterns.

Block 1:

2 convolution layers with 16 filters (3×3 kernel, ReLU, same padding)

1 max-pooling layer (2×2) → halves the spatial dimensions

Block 2:

2 convolution layers with 32 filters

1 max-pooling layer (2×2)

Block 3:

2 convolution layers with 64 filters

This block does not downsample further, acting as the bottleneck

These layers extract multi-scale hierarchical features from the input, helping the model learn vessel textures, edges, and branching structures at different scales.

🔼 Decoder (Expanding Path):
The decoder aims to reconstruct the segmented output by upsampling the feature maps and refining them using high-resolution information from the encoder via skip connections.

Block 1 (Upsampling + Merge):

Upsample the 64-filter output from the encoder

Concatenate with the corresponding 32-filter encoder output

2 convolution layers with 32 filters

Block 2 (Upsampling + Merge):

Upsample again

Concatenate with the corresponding 16-filter encoder output

2 convolution layers with 16 filters

This symmetrical decoder allows the network to recover spatial context lost during downsampling, crucial for precise vessel boundaries.

🟢 Final Output Layer:
A Conv2D layer with 1 filter and a (1×1) kernel is used to map each 16-channel pixel to a single-channel probability map.

A sigmoid activation function squashes the output between 0 and 1, representing the probability of each pixel belonging to a blood vessel.





![App Screenshot](https://raw.githubusercontent.com/bhaktii1802/Unet-Segmentation/main/micromachines-12-01478-g001.png)



## 🚀 What I Learned
- Built a custom U-Net architecture from scratch

- Learned how to combine multiple loss functions

- Handled image–mask alignment and grayscale preprocessing

- Tuned performance for real-world medical datasets

