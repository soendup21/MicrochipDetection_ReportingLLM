#Importing the libraries
import os
import time
import pickle
import numpy as np
import tensorflow as tf                                     # type: ignore
from tensorflow.keras.preprocessing import image            # type: ignore
#print(tf.__version__)

# Environment setup to disable GPU and set logging level
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model once when the program starts
model_path = 'CNN_code/best_rgb_chip_model_cate.keras'      # Provide the path to your saved model
model = tf.keras.models.load_model(model_path)              # Or use load_model(model_path)

# Load the class indices once
with open('CNN_code/chip_rgb_class_indices.pkl', 'rb') as f:   #Load class indices. Provide the path to your class indices .pkl file
    class_indices = pickle.load(f)
    print("Class indices:", class_indices)

#Making a single prediction 
#Deploying model on images
def make_prediction(image_path):
    # Preprocess the image
    predict_image = image.load_img(image_path, target_size=(224, 224))      #Load the image and resize it to 224x224 pixels (same size as used during training)
    predict_image = image.img_to_array(predict_image)       #Converts the loaded image from a PIL (Python Imaging Library) format to a NumPy array(predict method expect as input a 2D array)
    #Predict method has to be called on the exact same format that was used during the training, so we train with dataset in batch of 32 pictures then the input image must be in the batch also even though it is only 1 image
    predict_image = np.expand_dims(predict_image, axis=0)   #Adds an extra dimension to the image array, making it of shape (1, 224, 224, 3). This extra dimension corresponds to the batch size. Even though you are only predicting for one image, the model expects input in batches, so you need to include this dimension
    np.set_printoptions(precision=4)                        #set the show value to 4 digit
    
    #Process result
    result = model.predict(predict_image)                   #runs the predict_image through the CNN model and returns the prediction. The result will be a NumPy array where each element represents the predicted probability for each class. For example, if there are two classes, result might look something like [[0.1, 0.9]]
    
    # Find predicted class and label
    predicted_class_index = np.argmax(result, axis=1)[0]    #np.argmax(result, axis=1) returns the index of the highest probability along the axis corresponding to classes (axis=1 is look across row, axis=0 is look across column) then store in predicted_class_index. Since there's only one image, np.argmax(result, axis=1) returns an array with one element. After np.argmax return its value the [0] is used to extract this single value from matrix or array.
    prediction = None                                       # Initialize to None in case no match is found
    for class_label, index in class_indices.items():        #This loop goes through the training_set.class_indices dictionary and find its index. In this case training_set.class_indices={'chip': 0, 'empty': 1} training_set.class_indices.items()=[('chip', 0), ('empty', 1)] =>Class: chip, Index: 0,Class: empty, Index: 1 . So index in will return either 0 or 1
        if index == predicted_class_index:                  #checks which class label corresponds to the predicted_class_index
            prediction = class_label                        #Once it finds the correct class label, it assigns that label to the variable prediction
            break
    
    return prediction, result

# Example usage: Reuse the model for multiple predictions
image_path = 'CNN_code/chip_dataset/single_prediction/Bad_mark_from_BMS_mold_522.jpg'
start_time = time.time()
prediction, result = make_prediction(image_path)
print(f"Result: {result}")
print(f"Prediction: {prediction}")
print(f"Processing Time: {time.time() - start_time:.2f} seconds")