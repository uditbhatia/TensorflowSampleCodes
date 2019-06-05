import tensorflow as tf
from tensorflow import keras
import numpy as np


def train(x_train, y_train):

    model = tf.keras.Sequential([keras.layers.Dense(units=1,
                                                    input_shape=[1])])
    model.compile(optimizer="sgd",
                  loss="mean_squared_error")

    model.fit(x_train, y_train, epochs=500)

    return model


def predict(model, x_predict):

    return model.predict(x_predict)



def main():

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = train(x_train=xs,
          y_train=ys)
    print("Prediction: {}".format(model.predict([10.0])))


if __name__ == "__main__":
    main()