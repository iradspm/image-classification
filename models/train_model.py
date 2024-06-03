import warnings
import logging

warnings.filterwarnings("ignore")

import numpy as np

import tensorflow as tf

# metrics
from sklearn.metrics import classification_report

from data.process_data import (
    process_images
)
from model_architecture import (
    build_model
)


train_images, train_labels, test_images, test_labels = process_images()
model = build_model()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='/content/final_model.h5',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max', verbose=1,
        save_best_only=True)

history = model.fit(train_images, train_labels, epochs=10, callbacks=[model_checkpoint_callback],
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

logging.info(f"Model accuracy of test data: {test_acc}")

y_pred = np.argmax(model.predict(test_images), axis=-1)
# classification report
logging.info(classification_report(test_labels, y_pred, target_names = class_names))

