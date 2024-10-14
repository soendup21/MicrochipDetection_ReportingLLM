#%%                                                        #####Importing the libraries#####
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping                    # type: ignore #import Model EarlyStopping for automatically stop training when the validation loss stops improving
from tensorflow.keras.callbacks import ModelCheckpoint                  # type: ignore #import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # type: ignore #import ImageDataGenerator from image sub-module of the pre-processing module of the Keras library(from libraryName.moduleName.submoduleName import className)
#print(tf.__version__)                                                  #print the version of TensorFlow using

#%%                                                        #####Data Preparing and Preprocessing##### %%
#1st apply transformation on all images of training set to avoid overfitting(without transformation there will be a huge difference between accuracy on training set and test set)
#Image augmentation = A simple geometrical transformation or shifting pixels or zoom in&out or rotation or horizontal&vertical flip on image

#---Preprocessing the Training set---
train_datagen = ImageDataGenerator(
    rescale=1./255,                                     # Apply feature scaling to every pixels by dividing their value so from 0 to 255 will be 0 to 1 called Normalization
    shear_range=0.2,                                    # Randomly applies a shear transformation to the image, which is a form of geometric transformation that slants the shape of an object. This helps prevent overfitting by augmenting the training data
    zoom_range=0.2,                                     # Randomly zooms in or out on the image by up to 20%, helping to prevent overfitting by providing more varied training examples.
    horizontal_flip=True,                               # Randomly flips images horizontally to create augmented versions of the training data, again reducing overfitting by exposing the model to different perspectives of the images.
    rotation_range=40,                                  # Added rotation
    brightness_range=[0.8, 1.2])                        # Adjust brightness between 80% and 120%
training_set = train_datagen.flow_from_directory(       # Generates batches of augmented images directly from the directory containing the training set images.
    'dataset/chip_dataset/training_set',                        # Path to the directory where the training set images are stored.
    target_size=(224, 224),                             # Resize the image to 224x224 pixels. Reduce the computations of the machine,less computation intensive(final size of image fed to CNN)
    batch_size=32,                                      # Number of images per batch
    class_mode='categorical')                           # Can be binary or categorycal
#%%
#---Preprocessing the Test set---
test_datagen = ImageDataGenerator(                      # Use ImageDataGenerator object to apply the transformation to the test images but not apply same transformation as training set because we need to keep test images intact like original one to avoid information leakage from test set
    rescale=1./255)                                     # Rescale because we need to apply same scaling as traing set to CNN for future predict method
test_set = test_datagen.flow_from_directory(            # Connect train_datagen object to our test set images or import images (NameOfImportingSet = ObjectContainingClass.MethodofClass)
    'dataset/chip_dataset/test_set',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical')


#%%
#---Save class indices---    use because when save a model using model.save(), it does not include the class_indices (the mapping between class names and numerical labels).
with open('CNN_code/chip_rgb_class_indices.pkl', 'wb') as f:         # Opens a file named 'chip_class_indices.pkl' in 'wb' mode, which stands for "write binary."
    pickle.dump(training_set.class_indices, f)          # Pickle.dump() saves the object training_set.class_indices (a dictionary that maps class names to numbers) into the file f. ///The class_indices dictionary might look like {'cat': 0, 'dog': 1}///


#%%                                                        #####Building the CNN#####
#Initialising the CNN = CNN is sequence of layers, so we going to initialize CNN with the same class(sequential class)
cnn = tf.keras.models.Sequential()                      #Create a new instance of the Sequential model, which is a sequential model in Keras, where the CNN will store the model that can be further layered and trained (obj = tensorflow.library.models module.class)

#Input layer
cnn.add(tf.keras.layers.Input(shape=(224, 224, 3)))

#1st Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))       #add convolutional layer add(tf.keras.layers.Conv2D(filters=number of filters,kernel_size=number of Row by Col matric,activation='name of activation function',input_shape=[width value,height value,image dimension]) [lecturer find CNN architecture from online and classic one is 32 kernel]

#2nd Pooling Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))           #add pooling layer addadd(tf.keras.layers.MaxPool2D(pool_size=(row,col),strides=number of stride,padding='valid')) *padding is up to you valid or same but better for default

# Extra Dropout layer to prevent overfitting
cnn.add(tf.keras.layers.Dropout(0.25))  
        
#2.1 Adding second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))             #remove input_shape because it must entered only when adding first layer. To automatically connect first layer to input layer.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
cnn.add(tf.keras.layers.Dropout(0.25))  # Dropout layer to prevent overfitting

#2.2 Adding third Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))             #remove input_shape because it must entered only when adding first layer. To automatically connect first layer to input layer.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
cnn.add(tf.keras.layers.Dropout(0.25))  # Dropout layer to prevent overfitting

#3rd Flattening Layer
cnn.add(tf.keras.layers.Flatten())                                                      #this class dont need any parameters

#4th Fully Connection Layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))                             #add hidden layer add(tf.keras.layers.Dense(units=number of neurons,activations='name of avtivation function')) *you are dealing with complex problem(Computer Vision)so choose 128 *As long as you havent reached the output layer I would recommend to use a rectifier activation function
cnn.add(tf.keras.layers.Dropout(0.5))  # Dropout layer to prevent overfitting

#4.1 Adding second Fully Connection Layer
cnn.add(tf.keras.layers.Dense(units=64,activation='relu'))

#5th Output Layer
cnn.add(tf.keras.layers.Dense(units=3,activation='softmax'))                            #add output layer which is still be fully connected to previous hidden layer(code again the sense code) add(tf.keras.layers.Dense(units=number of output neurons,activation='sigmoid')) *use units=1 because we are doing binary classification *activation function for output layer is 'sigmoid for bianry classification' and 'softmax for multi-class classification'

##**************************************************************JUST CHANGE THE units=# in line 84 if have more classes


#%%                                                        #####Training the CNN#####
#1st Compiling the CNN
#Connect CNN to an optimizer, loss function and some metrics
#gonna choose adam optimizer to perform stochastic gradient descent to update the weight in order to reduce the loss error between the prediction and the target
#gonna choose binary cross entropy loss because doing binary classification task
#gonna choose accuracy matrics because the most relevant way to measure the performance of a classification model
cnn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])     #for optimizer='adam', the adam will be set to keras's default

#%%
#2nd Training the CNN on the Training set and evaluating it on the Testset *fit method always Training the CNN on the Training set

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)                               # Patience parameter specifies how many epochs to wait for improvement before stopping (e.g., patience=5 means if the val_loss doesn't improve for 5 epochs, training will stop)
checkpoint = ModelCheckpoint('CNN_code/best_rgb_chip_model_cate.keras', monitor='val_loss', save_best_only=True, mode='min')         # Save the best weight,validation
cnn.fit(training_set,validation_data=test_set,epochs=50, callbacks=[checkpoint, early_stopping])                        # Fit(x=Dataset use to train,validation_data=Dataset used to validate and evaluate the model's performance during training, Epochs=the number of times the model will train on the entire dataset. ////*The appropriate number of epochs can be determined by testing and gradually increasing the number, as choosing the right number of epochs is crucial. If the number of epochs is too low, the model might not learn enough (underfitting), but if it's too high, it could lead to overfitting
cnn.save('CNN_code/chip_rgb_model_cate.keras')                                                                                       # Save latest trained model

print("Class indices:", training_set.class_indices)     #prints the dictionary that maps class labels (like 'cat' and 'dog') to numerical indices (like 0 and 1)