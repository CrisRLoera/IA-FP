import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import tempfile
import os
import shutil
import base64

st.set_page_config(page_title="🔍 Mineral Recognition", layout="centered")

logo = Image.open("logo-r.png")
st.image(logo,width=150)

st.title("Mineral Recognition")

img_height = 180
img_width = 180

# Cargar modelo y clases
model_file = st.file_uploader("📁 Carga tu modelo `.keras`", type=["keras"])
classes_file = st.file_uploader("📁 Carga el archivo `class_names.pkl`", type=["pkl"])

if model_file and classes_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_class:
        tmp_class.write(classes_file.read())
        classes_path = tmp_class.name

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(classes_path, "rb") as f:
        class_names = pickle.load(f)

    st.success("✅ Modelo y clases cargadas correctamente.")

    uploaded_image = st.file_uploader("📸 Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="🖼 Imagen cargada", use_container_width=True)

        if st.button("🔍 Predecir"):

            tmp_dir = tempfile.mkdtemp()
            class_dir = os.path.join(tmp_dir, "dummy")
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, "img.jpg")
            image.save(image_path)


            ds = tf.keras.utils.image_dataset_from_directory(
                tmp_dir,
                image_size=(img_height, img_width),
                batch_size=1,
                shuffle=False
            )

            for images, _ in ds:
                preds = model.predict(images)
                break

            predicted_index = np.argmax(preds[0])
            predicted_class = class_names[predicted_index]
            confidence = 100 * np.max(preds[0])

            st.markdown(f"### 🧠 Predicción: **{predicted_class}**")
            st.markdown(f"📈 Confianza: **{confidence:.2f}%**")

            shutil.rmtree(tmp_dir)

else:
    st.info("📌 Carga tu modelo `.keras` y archivo `class_names.pkl` para comenzar.")
