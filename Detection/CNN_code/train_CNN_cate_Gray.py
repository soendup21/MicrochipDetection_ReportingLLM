#%%                                                        #####Importing the libraries#####
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator    # type: ignore

#%%                                                        #####Data Preparing and Preprocessing#####
# --- Preprocessing the Training set ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,   # Added rotation
    brightness_range=[0.8, 1.2]
)

training_set = train_datagen.flow_from_directory(
    'dataset/chip_dataset/training_set',
    target_size=(224, 224),
    color_mode='grayscale',    # Change color mode to grayscale
    batch_size=32,
    class_mode='categorical'
)

# --- Preprocessing the Test set ---
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/chip_dataset/test_set',
    target_size=(224, 224),
    color_mode='grayscale',    # Change color mode to grayscale
    batch_size=1,
    class_mode='categorical'
)

# --- Save class indices ---
with open('CNN_code/chip_gray_class_indices.pkl', 'wb') as f:
    pickle.dump(training_set.class_indices, f)

#%%                                                        #####Building the CNN#####
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Input layer
cnn.add(tf.keras.layers.Input(shape=(224, 224, 1)))   # Change input shape to accept grayscale (1 channel)

# 1st Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
cnn.add(tf.keras.layers.Dropout(0.25))

# 2nd Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
cnn.add(tf.keras.layers.Dropout(0.25))

# 3rd Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
cnn.add(tf.keras.layers.Dropout(0.25))

# Flattening Layer
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))

# Fully Connected Layer
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

#%%                                                        #####Training the CNN#####

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('CNN_code/best_gray_chip_model_cate.keras', monitor='val_loss', save_best_only=True, mode='min')

cnn.fit(training_set, validation_data=test_set, epochs=50, callbacks=[checkpoint, early_stopping])
cnn.save('CNN_code/chip_gray_model_cate.keras')

print("Class indices:", training_set.class_indices)