
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "bird_model.keras"
LABELS_PATH = PROJECT_DIR / "models" / "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as file:
    class_names = json.load(file)

image_path = input("Enter the path of a bird image: ").strip().strip('"')

image = load_img(image_path, target_size=(160, 160))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)
predicted_index = np.argmax(predictions[0])
predicted_species = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\nPredicted bird species:", predicted_species)
print(f"Model confidence: {confidence:.2f}%")
