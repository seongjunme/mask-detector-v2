from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data, IMG_SIZE

LR = 1e-4
EPOCHS = 20
batch_size = 50

x_data, y_data = load_data('dataset')

(x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data,
                                                  test_size=0.2, shuffle=True, 
                                                  random_state=123)

aug = ImageDataGenerator(
    rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

opt = Adam(learning_rate=LR, decay=LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
	aug.flow(x_train, y_train, batch_size=batch_size),
	steps_per_epoch=len(x_train) // batch_size,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // batch_size,
    epochs=EPOCHS)

model.save('model_2class.h5')