# Lungs-Cancer-Prediction-Using-ML

**Lung Cancer Type Prediction using SVM and PCA**

This project focuses on predicting lung cancer types—normal, benign, and malignant—using advanced machine learning techniques like Support Vector Machine (SVM) and Principal Component Analysis (PCA). The aim is to help in early detection and accurate classification, which is crucial for timely medical interventions.

**Project Overview:**
Lung cancer detection is a critical task in medical imaging. To make the diagnosis more efficient, this project leverages machine learning algorithms. The dataset contains three categories of lung images:

Normal: Images showing healthy lung tissues.
Benign: Images of non-cancerous, but abnormal lung growths.
Malignant: Images indicating the presence of cancerous cells.
**Key Steps in the Process:**
Preprocessing: The dataset was first preprocessed to ensure consistency and quality of the images.
Feature Engineering: Relevant features were extracted from the images to improve the model’s accuracy.
Dimensionality Reduction with PCA: PCA was applied to reduce the dimensionality of the dataset, speeding up the model and enhancing performance.
Classification using SVM: The Support Vector Machine (SVM) model was trained to classify the images into the three categories. SVM is known for its robustness in high-dimensional spaces.
User-Friendly Interface:
To make this tool accessible, a graphical user interface (GUI) was developed using Streamlit, allowing users to easily upload their lung images and receive predictions. Whether you're a medical professional or a researcher, this interface simplifies interaction with the model.

**How It Works:**
Upload an Image: Simply upload a lung scan image via the interface.
Get Prediction: The model will classify the image into one of the three categories (normal, benign, malignant).
Instant Results: With a quick prediction, users can get valuable insights in seconds.
