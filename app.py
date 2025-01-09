import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('model/cifar10_model.h5')

uploaded_file = st.file_uploader("Send a photo for analysis:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = np.array(image.resize((32, 32))) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_label = np.argmax(prediction)

    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    st.write(f"Prediction: {class_names[class_label]}")

