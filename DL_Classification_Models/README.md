# Deep Learning Classification Models

## Overview

This code implements a traditional classifier for breast cancer classification using ultrasound images. The classifier is designed to classify images into three categories: normal, benign, and malignant. The code is structured to be easily integrated into a larger segmentation and classification task.

## Dataset

The code uses the Breast Cancer Ultrasound Images Dataset from Hugging Face. The dataset consists of ultrasound images of breast tissues, labeled as normal, benign, or malignant.

## Classifier Architecture

The classifier uses a ResNet50 architecture, pre-trained on ImageNet. The architecture is modified to accommodate the specific requirements of the breast cancer classification task.

## Key Components

- Data Loading: The code loads the dataset from Hugging Face and organizes the data into training, validation, and testing sets.
- Data Preprocessing: The code preprocesses the images by resizing, normalizing, and converting them to tensors.
- Model Definition: The code defines the ResNet50 architecture and modifies it to accommodate the breast cancer classification task.
- Training: The code trains the model using the training set and evaluates its performance on the validation set.
- Evaluation: The code evaluates the model's performance on the testing set and calculates metrics such as accuracy, precision, recall, and F1 score.

## Use Cases

This code can be useful in a larger segmentation and classification task in the following ways:

- Further in Image Segmentation Task: The classifier can be used as a pre-processing step for image segmentation tasks, helping to identify regions of interest in the images. 

## Future Work ~ Improvements That Can Be Made

- Hyperparameter Tuning: The code can be modified to perform hyperparameter tuning to improve the model's performance.
- Ensemble Methods: The code can be modified to use ensemble methods, such as bagging or boosting, to improve the model's performance.
- Transfer Learning: The code can be modified to use transfer learning to adapt the model to other medical imaging tasks.


