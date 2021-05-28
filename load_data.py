from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

IMG_SIZE = 224


def load_data():
    img_paths = list(paths.list_images('dataset'))
    x_data = []
    y_labels = []

    for img_path in img_paths:
        label = img_path.split(os.path.sep)[-2]
   
        image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = preprocess_input(image)

        x_data.append(image)
        y_labels.append(label)

    x_data = np.array(x_data], dtype="float32")
    y_labels = np.array(y_labels)

    lb = LabelBinarizer()
    y_labels = lb.fit_transform(y_labels)
    y_labels = to_categorical(y_labels)
    
    return x_data, y_labels

