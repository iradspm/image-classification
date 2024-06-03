from keras import datasets

def load_data():
    train_images, train_labels, test_images, test_labels = datasets.cifar10.load_data()
    return train_images, train_labels, test_images, test_labels