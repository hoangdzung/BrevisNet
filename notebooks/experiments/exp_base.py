#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    help='datasete name')
parser.add_argument('model',
                    help='model name')
parser.add_argument('--cuda', default='0',
                    help='cuda')
parser.add_argument('--model_dir', default='./model',
                    help='directory to save models')
parser.add_argument('--seed', type=int, default=42,
                    help='seed number')
parser.add_argument('--epochs', type=int, default=50, 
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch_size')

args = parser.parse_args()

import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
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
from data_utils import load_dataset


model_dir = os.path.join(args.model_dir, '{}_{}'.format(args.dataset, args.model))
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
BASEMODEL_PATH = os.path.join(model_dir, '{}_{}_base_seed{}.hdf5'.format(args.dataset, args.model, args.seed))


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

train_ds, test_ds, validation_ds, meta_data = load_dataset(args.dataset, 224 if args.dataset == 'cifar10' else 96)


# ## From Scratch Model
model_to_class = {'alexnet': raw_models.alexnet.get_model,
                 'resnet': raw_models.resnet.get_model,
                 'inception': raw_models.inception.get_model,
                 'wideresnet': raw_models.wide_resnet_pretrain.get_model,
                 'nasnet': raw_models.nasnet.get_model,
                 'mobilenet': raw_models.mobilenetv2.get_model}
assert args.model in model_to_class
base_model = model_to_class[args.model](meta_data['input_size'], meta_data['n_classes'], meta_data['n_channels'], 'imagenet' if args.dataset == 'cifar10' else None)
base_model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')
base_model.fit(train_ds, epochs=args.epochs, validation_data = validation_ds, batch_size=args.batch_size,callbacks=[earlystop])
flops = get_branched_flops(base_model)
print(f"FLOPS: {flops:.03} G")
loss, accuracy = base_model.evaluate(test_ds, batch_size=args.batch_size)
print("Test accuray:", accuracy)
base_model.save(BASEMODEL_PATH)
print(base_model.summary())