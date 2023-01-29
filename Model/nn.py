import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),  # (256, 256)
        tf.keras.layers.Conv2D(128, (7, 7), activation='relu'),  # (250, 250)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (249, 249)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (243, 243)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),    # (248, 248)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (242, 242)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (241, 241)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (243, 243)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (248, 248)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (242, 242)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (241, 241)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (243, 243)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (248, 248)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (242, 242)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (241, 241)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (243, 243)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (248, 248)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),  # (242, 242)
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),  # (241, 241)
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model



def main():
    X = np.load("../Data/FoodPictures.npy") / 255.
    y = np.load("../Data/FoodNutrients.npy")
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69, shuffle=False)

    model = build_model()
    print(model.summary())
    model.compile(loss=tf.keras.losses.MeanSquaredError, optimizer=tf.keras.optimizers.Adam(0.01))
    model.fit(X_train, y_train, epochs=50, batch_size=16)

if __name__ == "__main__":
    main()