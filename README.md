# leaf-disease-image-detection

This repository implements a machine learning pipeline for detecting and classifying diseases in plant leaves using image processing and deep learning techniques. The project aims to assist in identifying plant diseases early, enabling effective intervention to improve crop yield and quality.

# Features
- Data Preprocessing
  - Handles image resizing, normalization, and augmentation for robust training.
  - Supports large datasets with diverse plant species and disease types.

- Deep Learning Models
  - Utilizes Convolutional Neural Networks (CNNs) for image classification.
  - Support for transfer learning using pre-trained models (e.g., ResNet, MobileNet, EfficientNet).

- Evaluation Metrics
  - Performance evaluated using metrics like accuracy, precision, recall, and F1-score.

- Real-Time Inference
  - Offers a lightweight API for deploying the trained model to predict diseases from leaf images.

# Getting Started
# Prerequisites
Ensure the following are installed on your system:

- Python 3.7+
- Required libraries: tensorflow, keras, numpy, pandas, opencv-python, matplotlib, and seaborn.
- GPU (optional but recommended for faster training).

# Installation Steps
Clone the Repository

# bash
    git clone https://github.com/BenedictIbe/leaf-disease-image-detection.git  
    cd leaf-disease-image-detection  

# Install Dependencies

# bash
    pip install -r requirements.txt  

# Prepare the Dataset
- Use publicly available datasets like PlantVillage.
- Place the dataset in the data/ folder with subdirectories for each class (e.g., Healthy/, Diseased/).
- Train the Model

Run the training script:

# bash
    python train_model.py  
- Test the Model
- Evaluate the trained model on test data:

# bash
    python test_model.py  

- Deploy the Model
Use the provided API or deploy the model to a cloud service for real-time predictions.

# Project Workflow
- Data Loading and Preprocessing
- Reads leaf images from the dataset.
- Applies preprocessing techniques like resizing, cropping, normalization, and augmentation.

# Model Training
- Trains a CNN-based model to classify leaf images into healthy or disease categories.
- Fine-tunes hyperparameters and optimizes the model for high accuracy.

# Model Evaluation
- Evaluates performance on a separate test set.
- Outputs detailed metrics and generates visualizations.

# Deployment
- Converts the trained model into a deployable format (e.g., TensorFlow SavedModel).
- Implements a REST API for real-time inference.

# Folder Structure

# bash
    leaf-disease-image-detection/  
    ├── data/                   # Dataset directory  
    │   ├── Healthy/            # Subfolder for healthy leaf images  
    │   ├── Diseased/           # Subfolder for diseased leaf images  
    ├── models/                 # Trained model files  
    ├── scripts/                # Python scripts for various stages of the project  
    │   ├── train_model.py      # Model training script  
    │   ├── test_model.py       # Model testing and evaluation script  
    │   └── preprocess.py       # Preprocessing utilities  
    ├── notebooks/              # Jupyter notebooks for exploration and visualization  
    ├── results/                # Evaluation results, metrics, and plots  
    ├── requirements.txt        # List of dependencies  
    └── README.md               # Project documentation  

# Usage Examples
# Training the Model
To train the model with default parameters, run:

# bash
    python train_model.py --epochs 50 --batch_size 32 --learning_rate 0.001  

# Testing the Model
Evaluate the trained model on test data:

# bash
    python test_model.py --test_dir ./data/test/ --model_path ./models/leaf_model.h5  

- Inference on a Single Image
- Predict the disease class for a single leaf image:

# bash
    python predict.py --image_path ./data/sample_leaf.jpg --model_path ./models/leaf_model.h5  

# Results and Visualizations
  - Accuracy: Achieves high accuracy for detecting and classifying common leaf diseases.
  - Confusion Matrix: Displays detailed class-wise performance.
  - Training Metrics: Visualizes loss and accuracy curves during training.

# Future Enhancements
  - Add support for additional plant species and diseases.
  - Implement edge AI deployment for mobile and IoT devices.
  - Integrate disease treatment recommendations based on predictions.

# Contributing
Contributions to enhance the model, add more datasets, or improve deployment workflows are highly welcome! Submit issues or pull requests for review.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contact
For questions, suggestions, or collaboration, feel free to reach out:

Author: Benedict Ibe
GitHub Profile: BenedictIbe
