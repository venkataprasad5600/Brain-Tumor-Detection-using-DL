# 📌  Brain-Tumor-Detection-using-DL

# 📝 Overview
This project aims to develop a machine learning model to detect brain tumors from MRI scans. The system processes MRI images, extracts relevant features, and classifies them as either benign or malignant. The project leverages deep learning techniques, particularly convolutional neural networks (CNNs), to accurately predict the presence of brain tumors.

# 🛠️ project work flow
1. Data Collection
Collected a dataset of brain MRI images from an open-source repository (e.g., Kaggle).
The dataset contains labeled MRI images categorized into "benign" and "malignant" classes.
2. Data Preprocessing
Resized images to a uniform size (e.g., 128x128 pixels).
Converted images to grayscale to reduce computational complexity.
Normalized pixel values to a range of [0, 1] for better model performance.
Split the dataset into training, validation, and testing sets (e.g., 70% for training, 15% for validation, and 15% for testing).
3.  Model Architecture
Used Convolutional Neural Networks (CNNs) for image classification.
The model consists of several convolutional layers followed by pooling layers and dense layers for classification.
Model architecture:

Input Layer: 128x128x3 (Resized and RGB images)
Convolutional Layers: Apply filters to detect features like edges, textures, etc.
Pooling Layers: Max pooling to reduce spatial dimensions.
Dense Layers: Flatten and fully connected layers for final classification.
Output Layer: Softmax activation for multi-class classification.
4.  Model Training
Trained the model using the training set with a batch size of 32.
Used Adam Optimizer for optimization and Categorical Cross-Entropy Loss for loss calculation.
Evaluated the model on the validation set to fine-tune hyperparameters.
5.  Model Evaluation
Tested the model on the test dataset to evaluate its accuracy, precision, recall, and F1 score.
Plotted a confusion matrix to visualize model performance.
6.  Model Improvement
Applied data augmentation techniques (e.g., rotations, zooming, and flipping) to improve generalization.
Fine-tuned the learning rate and epochs for optimal performance.
7.  Final Prediction
Once the model was trained, predictions were made on new MRI images to detect whether a tumor is present and its type (benign or malignant).


# 📊 Results
Accuracy: Achieved an accuracy of approximately 95% on the test dataset.
Precision & Recall: The model performed well in both precision and recall metrics, indicating that it effectively distinguishes between benign and malignant tumors.
Confusion Matrix: Visual representation of the model's classification performance, with accurate tumor detection and few misclassifications.
# 🔧 Usage
Prediction on a New Image: To predict whether a brain tumor is benign or malignant from an MRI image, use the following command:


# ✅ Conclusion
This project demonstrates the use of deep learning techniques, particularly Convolutional Neural Networks, to detect brain tumors from MRI scans. The model successfully classifies the images into benign and malignant categories, offering potential applications in medical diagnostics and assisting healthcare professionals in early tumor detection.
