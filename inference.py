import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pprint

model = tf.keras.models.load_model("FoodNutrientCalculator2")


def inference_on_image(path):
    im = Image.open(path)

    left = (im.size[0] - 512) // 2
    right = left + 512
    top = (im.size[1] - 512) // 2
    bottom = top + 512

    im = np.array(im.crop((left, top, right, bottom)).resize((256, 256)), dtype=np.float16)

    prediction = model.predict(im[np.newaxis])
    f = np.load("NormalizationFactors.npy.npz")
    factors = f['factors']
    names = f['names']

    unnormalized_predictions = np.empty(7, dtype=np.float32)
    for i in range(7):
        unnormalized_predictions[i] = prediction[0][i] * factors[i][1] + factors[i][0]

    pprint.pprint({nutrient: value for nutrient, value in zip(names, unnormalized_predictions)})


def main():
    inference_on_image("GatorCornerDesert.jpg")



if __name__ == "__main__":
    main()
