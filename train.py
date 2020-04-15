# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:44:25 2020

@author: Adrian Curic
"""

import argparse

parser = argparse.ArgumentParser(description='train model on fashion mnist dataset')
parser.add_argument('-s','--savefile', default = 'trained',
                    help = "filename to save model. Default 'trained'")
parser.add_argument('-l','--loadfile', default = None,
                    help = "filename to load model before training. Default: None, the model is trained from scratch")
parser.add_argument('-e','--epochs', default = 30, type = int,
                    help = 'Number of epochs to train, default 30')
parser.print_help()
args = parser.parse_args()
print(args)

# %%

import numpy as np
import lib_easymile as lem

x_train, y_train, ycat_train, x_test, y_test, ycat_test = lem.load_preprocess_data()

input_shape = x_train.shape[1:]
# Number of classes in the fashion mnist. There are 10 classes
num_classes = len(np.unique(y_train))
img_shape = (28,28,1) #image shape to be used by convolutional layer

# %%

# Load a partially trained model if argument given:
if args.loadfile is not None:
    model = lem.load_model(lem.param_output_dir + args.loadfile)
else:
    # Compile the model
    # conv model is better
    model = lem.MiniVGGNet_model(input_shape, img_shape, num_classes, show_summary = True)

# %%

# Train the model

epochs = args.epochs

filename = lem.param_output_dir + args.savefile
dataset = (x_train, ycat_train, x_test, ycat_test)
lem.train_model(model, dataset, filename, epochs = epochs)

print('Model trained and saved to %s' % filename)