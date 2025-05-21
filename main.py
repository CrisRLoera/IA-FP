import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Cargar modelo
model = tf.keras.models.load_model("mineral_classifier_model.h5")

# Lista de clases (puedes también cargarla desde un .pkl)
# Ejemplo:
# with open("class_names.pkl", "rb") as f:
#     class_names = pickle.load(f)
class_names = ['Amethyst', 'Azurite', 'Calcite', 'Fluorite', 'Galena', 'Malachite', 'Quartz']

# Configuración de la app
st.set_page_config(page_title="Clasificador de Minerales", layout="centered")
st.title("🧪 Clasificador de Minerales con CNN")

uploaded_file = st.file_uploader("📸 Sube una imagen de un mineral", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar
    img_height = 180
    img_width = 180
    img = image.resize((img_width, img_height))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predecir
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])

    # Mostrar resultado
    st.markdown(f"### 🧠 Predicción: **{predicted_class}**")
    st.markdown(f"Confianza: **{confidence:.2f}%**")
