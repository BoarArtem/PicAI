import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sqlite3
import io

model = load_model('model/cifar10_model.h5')

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB,
    prediction TEXT
)
''')
conn.commit()

def save_to_db(image_data, prediction):
    cursor.execute("INSERT INTO predictions (image, prediction) VALUES (?, ?)", (image_data, prediction))
    conn.commit()

uploaded_file = st.file_uploader("Send a photo for analysis:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_resized = image.resize((32, 32))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    class_label = np.argmax(prediction)
    predicted_class = class_names[class_label]

    st.write(f"Prediction: {predicted_class}")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    save_to_db(image_bytes.getvalue(), predicted_class)

st.write("### History of Predictions")
cursor.execute("SELECT id, prediction FROM predictions")
data = cursor.fetchall()
for record in data:
    st.write(f"ID: {record[0]}, Prediction: {record[1]}")

conn.close()
