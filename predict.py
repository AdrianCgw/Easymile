# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:58:44 2020

@author: Adrian Curic
"""

import argparse

parser = argparse.ArgumentParser(description='predict on fashion mnist dataset')
parser.add_argument('-m','--modelname', default = '_best_trained', 
                    help = "filename to load model. Default '_best_trained")
parser.add_argument('-d','--dataset', default = None, 
                    help = "X features to classify. Must be preprocessed. By default it classifies the x_test from fashion mnist")
parser.add_argument('-s','--savefile', default = 'predictions',
                    help = "filename to save predictions. Default 'predictions'")
parser.print_help()
args = parser.parse_args()
print(args)

# %%

import numpy as np
import lib_easymile as lem

# Read daset or load the **preprocessed** x features to predict
if args.dataset is None:
    x_train, y_train, ycat_train, x_test, y_test, ycat_test = lem.load_preprocess_data()
else:
    x_test = np.load(lem.param_output_dir + args.dataset + '.npy')

print('Samples to predict: %d' % len(x_test))

#save the preprocessed x samples. 
#Use this code to create a set of x samples and test the load preprocessed feature
if False:
    filename = lem.param_output_dir + 'x_to_predict'
    np.save(filename + '.npy', x_test[:5000])    

input_shape = x_test.shape[1:]
# Number of classes in the fashion mnist. There are 10 classes
num_classes = 10
img_shape = (28,28,1) #image shape to be used by convolutional layer

# %%

#  Predict classes, save the results

#Load pretrained model
model = lem.load_model(lem.param_output_dir + args.modelname)

filename = lem.param_output_dir + args.savefile
prediction = model.predict(x_test)
prediction_class = np.argmax(prediction, axis = 1) 
np.save(filename + '.npy', prediction_class)
print('Predictions saved to %s' % filename)