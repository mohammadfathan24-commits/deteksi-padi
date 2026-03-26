import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="rice_model.tflite")
interpreter.allocate_tensors()

# Ambil detail input & output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Judul
st.title("🌾 Deteksi Penyakit Daun Padi")
st.write("Ambil atau upload gambar daun padi untuk deteksi penyakit")

# Pilih metode input
option = st.radio("Pilih metode:", ["Upload", "Kamera"])

image = None

# Upload
if option == "Upload":
    image = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])

# Kamera
elif option == "Kamera":
    image = st.camera_input("Ambil foto daun padi")

# Jika ada gambar
if image is not None:
    img = Image.open(image)
    st.image(img, caption="Gambar", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img = np.array(img)

    # Pastikan RGB (hindari error channel)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data)
    confidence = np.max(output_data)

    # Hasil
    st.success(f"🌿 Hasil: {labels[pred]}")
    st.info(f"📊 Akurasi: {confidence*100:.2f}%")

    # Saran (opsional tapi keren)
    if labels[pred] == "Brown Spot":
        st.warning("💊 Saran: Gunakan fungisida seperti Mancozeb")
    elif labels[pred] == "Leaf Blast":
        st.warning("💊 Saran: Gunakan fungisida Tricyclazole")
    elif labels[pred] == "Bacterial Leaf Blight":
        st.warning("💊 Saran: Gunakan bakterisida & atur irigasi")
    elif labels[pred] == "Healthy":
        st.success("✅ Daun sehat, tidak perlu tindakan")