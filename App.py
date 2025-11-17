import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import os

# Load model
model = load_model('FV.h5')

# Labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Categorization
fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Static nutrition data
nutrition_data = {
    "Banana": "89 kcal (per 100g)",
    "Apple": "52 kcal (per 100g)",
    "Mango": "60 kcal (per 100g)",
    "Orange": "47 kcal (per 100g)",
    "Watermelon": "30 kcal (per 100g)",
    "Grapes": "69 kcal (per 100g)",
    "Carrot": "41 kcal (per 100g)",
    "Potato": "87 kcal (per 100g)",
    "Tomato": "18 kcal (per 100g)",
    "Cucumber": "15 kcal (per 100g)",
}


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class)
    res = labels[y]
    return res.capitalize()


def run():
    st.title("🍍 Fruit & Vegetable Classification 🍅")

    img_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img)

        os.makedirs("upload_images", exist_ok=True)

        save_path = os.path.join("upload_images", img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = prepare_image(save_path)

            if result in vegetables:
                st.info("*Category → Vegetable 🥕*")
            else:
                st.info("*Category → Fruit 🍎*")

            st.success(f"*Predicted: {result}*")

            calories = nutrition_data.get(result, "Calories info not available")
            st.warning(f"{calories}")


run()