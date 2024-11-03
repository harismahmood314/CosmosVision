# Galaxy Classification Using Convolutional Neural Network

This project is a Convolutional Neural Network (CNN) model developed to classify images of galaxies into different types based on their morphology. The model can distinguish between five types of galaxies:

- Irregular
- Merging
- Smooth/Round
- Spiral
- Edge-On

The project is implemented using TensorFlow and Keras, with the images preprocessed and the model evaluated on a test set.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing on New Images](#testing-on-new-images)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Project Overview
This project aims to classify different galaxy types using a deep learning approach. The CNN model is trained on preprocessed images, resized, and normalized to improve model performance.

## Dataset
The dataset consists of galaxy images labeled into five classes. The images are grayscale, resized to 128x128 pixels, and normalized to have values between 0 and 1.

## Preprocessing
The following preprocessing steps were applied:

- **Grayscale Conversion**: All images were converted to grayscale.
- **Normalization**: Image pixel values were scaled to the range [0, 1].
- **Resizing**: Each image was resized to 128x128 pixels.
- **Label Encoding**: Labels were simplified into five main categories.
- **Data Splitting**: The dataset was split into training, validation, and test sets.

## Model Architecture
The CNN model consists of three convolutional blocks followed by fully connected layers:

- **Convolutional Layers**: Three convolutional layers with increasing filters (32, 64, and 128), ReLU activation, and Batch Normalization.
- **Pooling Layers**: MaxPooling layers to reduce spatial dimensions.
- **Dropout Layers**: Dropout for regularization, with dropout rates increasing in each block.
- **Dense Layers**: Flattened output connected to Dense layers with ReLU and softmax activations for classification.

### Model Summary
- **Input Shape**: 128x128x1 (grayscale images)
- **Output**: Five classes with softmax activation.

## Training
The model was trained using the Adam optimizer and categorical cross-entropy loss. Early stopping was applied to prevent overfitting.

### Training Parameters
- **Epochs**: 10
- **Batch Size**: 32
- **Early Stopping**: Monitored validation loss with patience of 3 epochs.

## Evaluation
The model was evaluated on the test set, achieving an accuracy of approximately \[Add accuracy here after testing\]. The loss and accuracy were measured to ensure the model's robustness.

## Testing on New Images
A function `GalaxyPredict` is provided to test the model on new images. This function takes an image file, preprocesses it (grayscale, normalize, resize), and returns the predicted class.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-Image
- OpenCV
- PIL

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository.
2. Prepare the dataset and place it in the specified path.
3. Run the training script to train the model.
4. Use `GalaxyPredict` to classify new galaxy images.

### Example
To test a new image:

```python
GalaxyPredict('/path/to/your/image.jpeg')
```

