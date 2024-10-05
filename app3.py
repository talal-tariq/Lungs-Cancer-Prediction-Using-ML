
# import streamlit as st
# import cv2
# import numpy as np
# from skimage.feature import hog
# import joblib

# # Load the trained SVM model
# svm_model = joblib.load('my_model.pkl')  # Assuming you have saved the model using joblib

# def preprocess_image(image):
#     # Preprocess the image
#     # Resize, convert to grayscale, extract HOG features, etc.
#     resized_image = cv2.resize(image, (128, 128))  # Resize image to match model training
#     grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     hog_features = hog(grayscale_image, orientations=9, pixels_per_cell=(8, 8),
#                        cells_per_block=(2, 2), block_norm='L2-Hys')  # Extract HOG features
#     return hog_features.reshape(1, -1)  # Reshape to match model input

# def predict_image(image):
#     # Preprocess the image
#     preprocessed_image = preprocess_image(image)
    
#     # Make prediction using the model
#     prediction = svm_model.predict(preprocessed_image)
#     confidence = np.max(svm_model.decision_function(preprocessed_image))
    
#     return prediction, confidence

# # Streamlit app
# def main():
#     st.title('Lung Cancer Prediction from Image')

#     # Upload image
#     uploaded_file = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Read the image
#         image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
#         # Display the uploaded image
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         # Predict the class of the uploaded image
#         prediction, confidence = predict_image(image)
#         predicted_class = 'Lung Cancer' if prediction[0] == 1 else 'No Lung Cancer'
        
#         # Display prediction result
#         st.write(f"Predicted Class: {predicted_class}")
#         st.write(f"Confidence: {confidence}")

# if __name__ == '__main__':
#     main()


# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC

# # Load the trained PCA and SVM models
# pca = joblib.load('Pca_model.pkl')  # Assuming you have saved the PCA model using joblib
# svm_model = joblib.load('my_model.pkl')  # Assuming you have saved the SVM model using joblib

# def preprocess_image(image_path):
#     # Read the image from the file path
#     real_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Resize the image to 400x400 pixels to match the model training
#     resized_gray_image = cv2.resize(real_image, (400, 400))
    
    
#     # Flatten the image
#     flattened_image = resized_gray_image.flatten().reshape(1, -1)
    
#     # Apply PCA transformation using the loaded PCA model
#     pca_image = pca.transform([flattened_image])
    
#     return pca_image


# def predict_image(image):
#     # Preprocess the image
#     preprocessed_image = preprocess_image(image)
    
#     # Make prediction using the model
#     prediction = svm_model.predict(preprocessed_image)
#     confidence = np.max(svm_model.decision_function(preprocessed_image))
    
#     return prediction, confidence

# # Streamlit app
# def main():
#     st.title('Lung Cancer Prediction from Image')

#     # Upload image
#     uploaded_file = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Read the image
#         image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
         
#         if image is not None:
#             # Display the uploaded image
#             st.image(image, caption='Uploaded Image', use_column_width=True)
            
#             # Predict the class of the uploaded image
#             prediction, confidence = predict_image(image)  # Pass the image array
#             predicted_class = 'Lung Cancer' 
#             if prediction[0] == 0:
#                 predicted_class = 'No Lung Cancer'
#             elif prediction[0] == 1:
#                 predicted_class = 'Malignant Lung Cancer'
#             elif prediction[0] == 2:
#                 predicted_class = 'Benign Lung Cancer'
            
#             # Display prediction result
#             st.write(f"Predicted Class: {predicted_class}")
#             st.write(f"Confidence: {confidence}")
#         else:
#             st.write("Error: Unable to read the uploaded image. Please try again.")
#     else:
#         st.write("Please upload an image file.")


# if __name__ == '__main__':
#     main()

import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load the trained models
pca = joblib.load('Pca_model.pkl')
svm = joblib.load('my_model.pkl')

# Categories
categories = ['Normal cases', 'Malignant cases', 'Bengin cases']

# Streamlit UI
st.title("Lung Cancer Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (400, 400))   # Resize to the same size as training images
    img_flatten = img.flatten().reshape(1, -1)
    
    # Apply PCA
    img_pca = pca.transform(img_flatten)
    
    # Predict using SVM
    prediction = svm.predict(img_pca)
    
    # Display the result
    st.write(f"Prediction: {categories[prediction[0]]}")