import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import PIL



batch_size=100
Data_dir='/home/louay/Desktop/ML/dump'

Data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=0.3)

train_generator = Data_gen.flow_from_directory(
    Data_dir,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)






model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




history = model.fit(train_generator, epochs=25, verbose = 1, validation_steps=3)




validation_generator = Data_gen.flow_from_directory(
    Data_dir,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation')



loss_and_metrics = model.evaluate(validation_generator)

#model.save('3_fing_v3.h5')






