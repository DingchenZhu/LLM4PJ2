import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)

img = Image.open("test_example.jpeg").convert("RGB").resize((150, 150))
arr = np.expand_dims(np.array(img) / 255.0, axis=0)

prob = model.predict(arr)[0][0]
print("肺炎概率：", prob)
