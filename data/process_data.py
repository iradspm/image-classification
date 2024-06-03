import warnings

warnings.filterwarnings("ignore")

import numpy as np

from PIL import Image, ImageEnhance

from load_data import (
    load_data
)

# pre-process images
def process_images():
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # image pre-processing
    train_images, test_images = train_images / 255.0, test_images / 255.0
    for i in range(len(train_images)):
        img = Image.fromarray((train_images[i] * 255).astype(np.uint8))
        # Adjust brightness
        brightness_factor = 1.2
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        # Adjust contrast
        contrast_factor = 1.5
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        train_images[i] = np.array(img) / 255.0

    for i in range(len(test_images)):
        img = Image.fromarray((test_images[i] * 255).astype(np.uint8))
        # Adjust brightness
        brightness_factor = 1.2
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        # Adjust contrast
        contrast_factor = 1.5
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        test_images[i] = np.array(img) / 255.0
        
    return train_images, train_labels, test_images, test_labels

