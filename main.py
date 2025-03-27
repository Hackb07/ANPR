import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add other languages if needed

st.title("📸 Live NPR App with EasyOCR")
st.write("Capture an image using the cam and extract the Info with NPR")

# Capture image from webcam
img_file_buffer = st.camera_input("Capture Image")

if img_file_buffer is not None:
    # Convert the image to a format suitable for processing
    image = Image.open(img_file_buffer)
    image_np = np.array(image)  # Convert to NumPy array

    # Display the captured image
    st.image(image, caption="Captured Image", use_column_width=True)

    # Perform OCR
    st.write("🔍 Extracting text from image...")
    results = reader.readtext(image_np)

    # Display extracted text
    st.subheader("Extracted Text:")
    extracted_text = "\n".join([res[1] for res in results])
    st.text_area("OCR Result:", extracted_text, height=200)

    # Download button for extracted text
    st.download_button("📥 Download Text", extracted_text, file_name="extracted_text.txt")

    # Display detected text with confidence scores
    st.subheader("Detected Text with Confidence Scores:")
    for (bbox, text, prob) in results:
        st.write(f"📌 `{text}` (Confidence: {prob:.2f})")
