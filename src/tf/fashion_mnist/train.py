from tensorflow import keras
import tensorflow as tf

def load_data():
    print("Loading Fashion Mnist dataset")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(
        "Fashion Mnist dataset loaded with, train_images: {}, train_labels: {}, test_images: {}, test_labels: {}".format(
            train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))
    return (train_images, train_labels), (test_images, test_labels)

def build():

    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()

    print("Starting training fashion mnist")
    model = build()
    model.fit(train_images,train_labels, epochs=5)

    print("Evaluate model:")
    model.evaluate(test_images,test_labels)


if __name__ == "__main__":
    main()