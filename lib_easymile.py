# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:04:54 2020

@author: Adrian Curic
"""

import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, BatchNormalization
from keras.regularizers import l2
from keras.datasets import fashion_mnist
from keras.models import model_from_json

from sklearn.model_selection import StratifiedKFold

'''
K-cross validation:
Accuracy: 0.9056 (+- 0.0032)

Pretrained model on x_train, saved as _best_trained 
Test Accuracy: 0.9021
'''

# %%

# Parameters ############################################################################
param_output_dir = 'output/' 

# %%
# Functions #############################################################################

def load_mnist(path, kind='train'):
    '''   Loads MNIST data from `path`

    Uses the reloaded mnist-fashion dataset available in the Zalando github
    https://github.com/zalandoresearch/fashion-mnist
    
    Function copied from Zalando github
    https://github.com/zalandoresearch/fashion-mnist
    '''
    
    import os
    import gzip
    import numpy as np

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def FC_model(input_shape, num_classes, dropout = 0, show_summary = False):
    ''' Create a simple neural network based on FC layers with 
    relu activation. Final layer: softmax classification
    '''
    
    # Set L2 regularisation
    reg = l2(0.001)
    
    model = Sequential()
    img_shape = (np.prod(input_shape),)
    model.add(Reshape(img_shape, input_shape = input_shape))

    model.add(Dense(512, activation='relu', activity_regularizer=reg))
	# BatchNormalization does not improve performance
    #model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', activity_regularizer=reg))
	#model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', activity_regularizer=reg))
	#model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))    
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics=['accuracy'])
    return model


def MiniVGGNet_model(input_shape, img_shape, num_classes, dropout = 0, show_summary = False):
    ''' Create a MiniVGGNet based on 2x (2x(conv,relu),pooling) + 3x FC (dense relu)
    '''
    # Set L2 regularisation
    reg = l2(0.001)
    
    # BatchNormalization not used as it does not improve results
    #Using channel last img format
    #chanDim = -1
    
    model = Sequential()
    model.add(Reshape(img_shape, input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activity_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activity_regularizer=reg))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activity_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activity_regularizer=reg))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(512, activation = 'relu', activity_regularizer=reg))
    #model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(256, activation = 'relu', activity_regularizer=reg))
    #model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(128, activation = 'relu', activity_regularizer=reg))
    #model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))    
    
    if show_summary:
        model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics=['accuracy'])
    return model

def save_model(model, filename):
    ''' Save model architecture and weights
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights( filename + '.h5')
   
def load_model(filename, compile_model = True):
    ''' Load model architecture and weights
    Compile model after loading (default)
    '''
    try:
        json_file = open(filename +'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(filename + '.h5')
    except FileNotFoundError as e:        
        print('Model save not found. Train and save a model first')
        raise e
        
    # model needs to be compiled before training, evaluation. 
    #The compile arguments might be different, can use **kwards for a general function
    # or compile seperately after loading
    if compile_model:
        loaded_model.compile(loss='categorical_crossentropy',
                      optimizer = 'rmsprop',
                      metrics=['accuracy'])
    return loaded_model

def load_preprocess_data():
    ''' Loads and preprocesses fashin mnist dataset
    Output: x_train, y_train, ycat_train, x_test, y_test, ycat_test
    '''
    # Load data using keras methods. Takes a long time, use preloaded data instead 
    #(x_train, y_train),  (x_test, y_test) = fashion_mnist.load_data()
    
    # Load pre-loaded data from Zalando github project
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
    
    # Preprocess data
    # It is an equivalent MinMax scaling as the input data is uint8
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # Number of classes in the fashion mnist. There are 10 classes
    num_classes = len(np.unique(y_train))
    
    # Handle categorical data
    # convert class vectors to binary class matrices
    ycat_train = keras.utils.to_categorical(y_train, num_classes)
    ycat_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, ycat_train, x_test, y_test, ycat_test

def train_model(model, dataset, filename = None, 
        dropout = 0,
        batch_size = 128,
        epochs = 30,
        ):
    ''' Trains the model and saves it to <filename>, if filename given
    Input:
        dataset = (x_train, ycat_train, x_test, ycat_test)
    '''
    x_train, ycat_train, x_test, ycat_test = dataset
    
    model.fit(x_train, ycat_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, ycat_test))
    if filename is not None:
        save_model(model, filename)

def evaluate_model(model, dataset):
    ''' Evaluates a trained model
    Input:
        dataset = (x_test, ycat_test)
    '''
    
    x_test, ycat_test = dataset
    score = model.evaluate(x_test, ycat_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

def cross_validate(model, dataset, filename,
        n_splits = 7,
        dropout = 0,
        batch_size = 128,
        epochs = 30,
        ):
    
    ''' Cross validates the model
    Saves the best trained model to filename
    Input:
        dataset = (x_train, y_train, ycat_train, x_test, y_test, ycat_test)
    '''

    # Save the untrained model. Each k validation will begin with an untrained model
    untrained_filename = param_output_dir + '_untrained_model.h5'
    model.save_weights(untrained_filename)
    
    x_train, y_train, ycat_train, x_test, y_test, ycat_test = dataset

    # Concatenate the train and test dataset. K fold will do the train/test split    
    # We need numerical y labels (y_all) for the Stratified split
    # and categorical y labels (ycat_all) for the train/tesst
    x_all = np.concatenate((x_train, x_test), axis = 0)
    y_all = np.concatenate((y_train, y_test), axis = 0)
    ycat_all = np.concatenate((ycat_train, ycat_test), axis = 0)
        
    accuracy_list = []
    loss_list = []
    
    skf = StratifiedKFold(n_splits = n_splits) 
   
    # Train the model for every split in StratifiedKFold
    for i, (idx_train, idx_test) in enumerate(skf.split(x_all, y_all)):
        # Load the untrained model weights 
        model.load_weights(untrained_filename)
        model.fit(x_all[idx_train], ycat_all[idx_train],
                        batch_size=batch_size,epochs=epochs,verbose=1)
        loss,accuracy = model.evaluate(x_all[idx_test], ycat_all[idx_test])
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        print('Fold %s,  %s: %.4f %s: %.4f' % (i, model.metrics_names[0], loss,
                                               model.metrics_names[1], accuracy))
        #Save the best model
        if accuracy == np.max(accuracy_list):
            print('Saving model')
            save_model(model, filename)
        
    print('Accuracy: %.4f (+- %.4f)' % (np.mean(accuracy_list), np.std(accuracy_list)))
    return accuracy_list  

def multi_train_model(dataset, best_filename, train_attempts = 7,
                      dropout = 0,
                      batch_size = 128,
                      epochs = 30,
                      ):
    '''
    Train the model several times, saves the best accuracy model to <best_filename>
    Input:
        dataset = (x_train, ycat_train, x_test, ycat_test)
        train_attemps = how many train attempts are performed
    '''
    eval_dataset = (dataset[2], dataset[3])
    
    # train the network several times, saves the best model:
    best_accuracy = 0
    for _ in range(train_attempts):
        # Random initialized model
        model = MiniVGGNet_model(input_shape, img_shape, num_classes, show_summary = False)
        train_model(model, dataset, filename = None, epochs = epochs, 
                    dropout = dropout, batch_size = batch_size)
        score = evaluate_model(model, eval_dataset)
        # Save model if the accuracy is better:
        if score[1] > best_accuracy:
            best_accuracy = score[1]
            save_model(model, best_filename)
            print('Saving model. Best_accuracy: %.4f' % best_accuracy)

# %%

# Main Code ############################################################################

if __name__ == '__main__':
    
    # Read daset
    x_train, y_train, ycat_train, x_test, y_test, ycat_test = load_preprocess_data()
    
    input_shape = x_train.shape[1:]
    # Number of classes in the fashion mnist. There are 10 classes
    num_classes = len(np.unique(y_train))
    img_shape = (28,28,1) #image shape to be used by convolutional layer
    
# %%

    # Compile the model
    
    # conv model is better
    model = FC_model(input_shape, num_classes, show_summary = True)
        
    model = MiniVGGNet_model(input_shape, img_shape, num_classes, show_summary = True)
    
# %%
    
    # Train the model
    
    # Recommended epochs for a full training:
    epochs = 30 
    # For a quick test of the code:
    epochs = 2 
    
    filename = param_output_dir + 'trained' 
    dataset = (x_train, ycat_train, x_test, ycat_test)
    train_model(model, dataset, filename, epochs = epochs)
    
    
# %%

    # Check model accuracy
    
    filename = param_output_dir + '_best_trained'
    
    # Can use best trained model from multiple training attempts:
    #filename = param_output_dir + '_best_trained'
        
    model = load_model(filename)
    dataset = (x_test, ycat_test)
    evaluate_model(model, dataset)

# %%
    
    #  Predict classes, save the results
    
    filename = param_output_dir + 'predictions'
    prediction = model.predict(x_test)
    prediction_class = np.argmax(prediction, axis = 1) 
    np.save(filename + '.npy', prediction_class)
    
    load_p = np.load(filename + '.npy')
    
# %%    

    # Perform cross validation, save the best trained network
    # Cannot use the cross best trained with with the orignal test data as it might have 
    # been trained on the test data !
    
    # Recommended for full k cross validation
    epochs = 30
    n_splits = 7
    # For a quick test of the code
    #epochs = 2
    #n_splits = 3
    filename = param_output_dir + '_s' 
    
    dataset = (x_train, y_train, ycat_train, x_test, y_test, ycat_test)
    accuracy_list = cross_validate(model, dataset, filename, n_splits = n_splits, epochs = epochs)

# %%

    # Train model several times, saves the best accuracy model
    epochs = 30
    train_attempts = 7
    # For a quick test:
    #epochs = 2
    #train_attempts = 7
       
    filename = param_output_dir + '_best_trained'
    dataset = (x_train, ycat_train, x_test, ycat_test)
    multi_train_model(dataset, filename, train_attempts = train_attempts , 
                      epochs = epochs)
