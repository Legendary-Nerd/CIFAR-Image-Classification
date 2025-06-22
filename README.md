# CIFAR Image Classification

## Overview

This repository implements a deep learning model for classifying images from the CIFAR-10 dataset. The model uses [CNNs (Convolutional Neural Networks)](https://en.wikipedia.org/wiki/Convolutional_neural_network) to achieve high performance on this image classification task.

CIFAR-10 contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal is to train a machine learning model to accurately predict the class of images based on their pixel values.

## Dataset

* The CIFAR-10 dataset consists of 10 different categories:

  * Airplane
  * Automobile
  * Bird
  * Cat
  * Deer
  * Dog
  * Frog
  * Horse
  * Ship
  * Truck

* The dataset is divided into:

  * 50,000 training images
  * 10,000 test images

### Download the Dataset

The CIFAR-10 dataset can be easily downloaded via `torchvision` or manually from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Features

* Preprocessing the CIFAR-10 dataset for training.
* Training a Convolutional Neural Network (CNN) model on CIFAR-10.
* Evaluating the model's performance using accuracy and loss metrics.
* Saving and loading trained models for future use.

## Requirements

The following dependencies are required for the project:

* Python 3.x
* PyTorch
* NumPy
* Matplotlib (for visualization)
* torchvision
