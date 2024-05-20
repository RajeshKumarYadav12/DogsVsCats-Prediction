import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained model
model_path = 'dog_vs_cat_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file not found: {model_path}")

# Define the class labels
class_labels = ['Cat', 'Dog']

# Streamlit app title
st.title("Dog vs Cat Image Classifier")

# Instructions
st.markdown("""
    **Instructions:**
    - Upload an image of a dog or a cat.
    - The model will classify the image and display the result along with the confidence level.
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_image(image):
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Resize the image to match model input
        test_img = cv2.resize(img, (256, 256))

        # Normalize the image
        test_img = test_img / 255.0

        # Prepare the image for prediction
        test_input = test_img.reshape((1, 256, 256, 3))

        # Make prediction
        prediction = model.predict(test_input)
        predicted_class_index = int(prediction[0][0] > 0.5)  # Assuming sigmoid activation
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][0] if predicted_class == 'Dog' else 1 - prediction[0][0]

        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Make prediction
    predicted_class, confidence = predict_image(uploaded_file)

    if predicted_class is not None:
        # Display the prediction as a heading
        st.subheader(f"This is a {predicted_class} Image.")

        # Display the image with caption
        st.image(uploaded_file, caption=f'{predicted_class} Image with Confidence {confidence:.2f}', use_column_width=True)
        st.write("")

        # Display the prediction details
        st.subheader("Prediction Details")
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")

# To run the app, use: streamlit run app.py
