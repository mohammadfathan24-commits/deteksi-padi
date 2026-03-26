import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="rice_model.tflite")
interpreter.allocate_tensors()

# Ambil detail input & output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Judul app
st.title("🌾 Deteksi Penyakit Daun Padi")

#kamera
camera = st.camera_input("Ambil foto langsung")

if camera is not None:
    image = Image.open(camera)

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar daun padi", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img)

    # Normalisasi (0-1)
    img = img.astype(np.float32) / 255.0

    # Tambah dimensi batch
    img = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Jalankan model
    interpreter.invoke()

    # Ambil hasil
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Ambil label tertinggi
    pred_index = np.argmax(output_data)
    confidence = np.max(output_data)

    # Tampilkan hasil
    st.success(f"Hasil: {labels[pred_index]}")
    st.info(f"Akurasi: {confidence*100:.2f}%")
