from tensorflow import keras

def load_data():
    print("Loading Fashion Mnist dataset")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(
        "Fashion Mnist dataset loaded with, train_images: {}, train_labels: {}, test_images: {}, test_labels: {}".format(
            train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))

def main():
    load_data()

    print("Starting training fashion mnist")

if __name__ == "__main__":
    main()