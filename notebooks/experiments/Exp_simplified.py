#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("../..") # Adds higher directory to python modules path.

import tensorflow as tf
from tensorflow import keras
from keras_flops import get_flops 
import numpy as np

import brevis
from brevis import raw_models
from utils import Brevis_loss_final, growth_update, branch_conv2d, save_outputs
from utils import getPredictions_Energy, infer_result, infer_result_OOD, get_branched_flops
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./model'
                    help='directory to save models')
parser.add_argument('--dataset'
                    help='datasete name')
parser.add_argument('--model',
                    help='model name')
parser.add_argument('--seed', type=int, default=42.
                    help='seed number')

args = parser.parse_args()

model_dir = os.path.join(args.model_dir, '{}_{}'.format(args.dataset, args.model))
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
BASEMODEL_PATH = os.path.join(model_dir, '{}_{}_base.hdf5')


# ## Model Determinism
# If using Deterministic Operations, enable the following block to ensure that at each step the model begins with the same random seed.
# This is particularlly helpful to directly compare hyperparameter changes for small improvements without the need to repeat multiple tests to get average results.
# It is slightly slower but still fairly fast.

_DETERMINISTIC = False
# _DETERMINISTIC = True

_SEED = args.seed
if _DETERMINISTIC: 
    import random
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    _SEED = args.seed

def reset_seeds(seed = _SEED):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


# ## Load dataset
# Datasets are loaded in batches of 32, input sizes of (32,32) and shuffled between epochs.
# For Cifar datasets, The standard train and set sizes are used, with a separate 5k training images separated for validation set purposes.

# In[4]:


INPUT_SIZE = 224
N_CHANNELS = 3
N_CLASSES = 10
DATASET = tf.keras.datasets.cifar10.load_data()
train_ds, test_ds, validation_ds = brevis.dataset.prepare.dataset(tf.keras.datasets.cifar10.load_data(),
                                                                32,5000,shuffle_size=15000,input_size=(INPUT_SIZE,INPUT_SIZE),
                                                                include_targets=False,num_outputs = N_CLASSES,reshuffle=True)

base_model = raw_models.alexnet.get_model(INPUT_SIZE, N_CLASSES, N_CHANNELS)


base_model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')

base_model.fit(train_ds, epochs=20, validation_data = validation_ds, batch_size=64,callbacks=[earlystop])

base_model.summary()
base_model.save(BASEMODEL_PATH)
