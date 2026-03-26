import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="rice_model.tflite")
interpreter.allocate_tensors()

# Ambil detail input & output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Judul aplikasi
st.title("🌾 Deteksi Penyakit Daun Padi")
st.write("Upload atau ambil foto daun padi untuk dideteksi")

# Pilih metode input
option = st.radio("Pilih metode input:", ["Upload Gambar", "Kamera"])

image = None

# Upload gambar
if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar daun padi", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Kamera langsung
elif option == "Kamera":
    camera = st.camera_input("Ambil foto daun padi")
    if camera is not None:
        image = Image.open(camera)

# Jika ada gambar
if image is not None:
    st.image(image, caption="Gambar", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img)

    # Pastikan 3 channel (RGB)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Input ke model
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output_data)
    confidence = float(np.max(output_data))

    # Hasil
    st.success(f"Hasil: {labels[pred_index]}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    # Saran sederhana
    if "blast" in labels[pred_index].lower():
        st.warning("Saran: Gunakan fungisida berbahan aktif Tricyclazole")
    elif "brown" in labels[pred_index].lower():
        st.warning("Saran: Gunakan fungisida Mancozeb")
    elif "healthy" in labels[pred_index].lower():
        st.success("Tanaman sehat 👍")
