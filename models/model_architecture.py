import tensorflow as tf

from keras import layers, models

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, padding = "same", activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, 3, padding = "same", activation='relu'))
    model.add(layers.Conv2D(64, 3, activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation = "softmax"))
    model.summary()

    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model
