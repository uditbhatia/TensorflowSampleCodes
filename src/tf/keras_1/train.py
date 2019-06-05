import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import os

def train(x_train, y_train):

    model = tf.keras.Sequential([keras.layers.Dense(units=1,
                                                    input_shape=[1])])
    model.compile(optimizer="sgd",
                  loss="mean_squared_error")

    model.fit(x_train, y_train, epochs=500)

    return model


def predict(model, x_predict):

    return model.predict(x_predict)

def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    print("Model successfully saved at: {}".format(output))
    return


def main(model_dir):

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = train(x_train=xs,
          y_train=ys)

    print("Prediction: {}".format(model.predict([10.0])))
    save_model(model, model_dir)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--model_output_dir', type=str)

    args, _ = parser.parse_known_args()
    main(model_dir=args.model_output_dir)