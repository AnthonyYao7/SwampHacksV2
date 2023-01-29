import tensorflow as tf
import numpy as np
from PIL import Image
import pprint
model = tf.keras.models.load_model("FoodNutrientCalculator2")


def inference_on_image(path):
    im = Image.open(path)
    im = np.array(im.resize((256, 256)), dtype=np.float16)

    prediction = model.predict(im[np.newaxis])
    f = np.load("NormalizationFactors.npy.npz")
    factors = f['factors']
    names = f['names']

    unnormalized_predictions = np.empty(7, dtype=np.float32)
    for i in range(7):
        unnormalized_predictions[i] = prediction[0][i] * factors[i][1] + factors[i][0]

    return {nutrient: value for nutrient, value in zip(names, unnormalized_predictions)}


def main():
    pprint.pprint(inference_on_image("TestData/lasagna.jpg"))


if __name__ == "__main__":
    main()
