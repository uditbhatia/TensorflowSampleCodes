from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.models import Model
import argparse
import os
import tensorflow as tf


def build():

    vgg19 = keras.applications.vgg19.VGG19(include_top=True,
                                           weights='imagenet',
                                           input_tensor=None,
                                           input_shape=None,
                                           pooling=None)
    print("Vgg19 model summary: {}".format(vgg19.summary()))

    layer_name = 'fc2'
    vgg19_ie_model = Model(inputs=vgg19.input,
                            outputs=vgg19.get_layer(layer_name).output)

    print("Vgg19 IE model summary: {}".format(vgg19_ie_model.summary()))

    return vgg19_ie_model


def predict(model, img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    print("Input Image shape: {}".format(img_data.shape))
    vgg19_feature = model.predict(img_data)

    print("Image feature vector generated with shape: {}".format(vgg19_feature.shape))

    return vgg19_feature

def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, output)
    print("Model successfully saved at: {}".format(output))


def main(img_path, model_out_dir):

    model = build()
    print("Image Feature Vector: {}".format(predict(model,img_path)))
    save_model(model, model_out_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_out_dir', type=str, required=True)

    args, _ = parser.parse_known_args()

    main(img_path=args.img_path,
         model_out_dir=args.model_out_dir)