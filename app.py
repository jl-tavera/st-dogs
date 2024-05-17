import streamlit as st
import numpy as np
import requests
from PIL import Image
import json
import time

# Google Cloud Function URL
GCF_URL = 'https://us-central1-operating-braid-423516-s3.cloudfunctions.net/predict'  # Replace with your GCF URL

st.title("Dog Breed Classifier")
st.write("Upload an image of a dog, and the model will predict the breed.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def array_to_string(array):
    array_str = ""	
    for row in array:
        for pixel in row:
            for channel in pixel:
                array_str += str(channel) + ","
    return array_str[:-1]


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the image to bytes
    image = np.array(
        Image.open(uploaded_file).convert("RGB").resize((256, 256)) # image resizing
    )
    image = image/255.0
    image = image.tolist()

    array_str = array_to_string(image)
    form_data = {"array": array_str}

    # Send the image to the Google Cloud Function
    response = requests.post(GCF_URL, data=form_data)

    # Parse the response from the Cloud Function
    if response.status_code == 200:
        result = response.json()
        st.write(f"Predicted Breed: {result['class']}")
        st.write(f"Confidence: {result['confidence']}%")
    else:
        st.write("Error in classification")