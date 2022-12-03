import os, sys
sys.path.append("../..") # Adds higher directory to python modules path.

import tensorflow as tf
import brevis

import medmnist
from medmnist import INFO, Evaluator
import dataset_without_pytorch
import numpy as np 

def load_dataset(dataset_name, input_size, is_mobile_model):
    if dataset_name == 'cifar10':
        N_CLASSES = 10
        N_CHANNELS = 3
        train_ds, test_ds, validation_ds = brevis.dataset.prepare.dataset(tf.keras.datasets.cifar10.load_data(),
                                                                32,5000,shuffle_size=15000,input_size=(input_size,input_size),
                                                                include_targets=False,num_outputs = N_CLASSES,reshuffle=True)
        meta_data = {'input_size': input_size, 'n_channels': N_CHANNELS, 'n_classes' : N_CLASSES}
        
    elif dataset_name == 'cifar100':
        N_CLASSES = 100
        N_CHANNELS = 3
        train_ds, test_ds, validation_ds = brevis.dataset.prepare.dataset(tf.keras.datasets.cifar100.load_data(label_mode="fine"),
                                                                32,5000,shuffle_size=15000,input_size=(INPUT_SIZE,INPUT_SIZE),
                                                                include_targets=False,num_outputs = 100,reshuffle=True)

    elif dataset_name.endswith('mnist'):
        info = INFO[dataset_name]
        task = info['task']
        N_CHANNELS = info['n_channels']
        N_CLASSES = len(info['label'])

        DataClass = getattr(dataset_without_pytorch, info['python_class'])

        try:
            train_dataset = DataClass(split='train', download=True)
        except:
            train_dataset = DataClass(split='train', download=True)        
        val_dataset = DataClass(split='val', download=True)
        test_dataset = DataClass(split='test', download=True)
        
        if N_CHANNELS == 1:
            if is_mobile_model:
                train_dataset.imgs = np.stack([train_dataset.imgs, train_dataset.imgs, train_dataset.imgs], -1)
                val_dataset.imgs = np.stack([val_dataset.imgs, val_dataset.imgs, val_dataset.imgs], -1)
                test_dataset.imgs = np.stack([test_dataset.imgs, test_dataset.imgs, test_dataset.imgs], -1)
                N_CHANNELS = 3
            else:
                train_dataset.imgs = np.expand_dims(train_dataset.imgs, -1)
                val_dataset.imgs = np.expand_dims(val_dataset.imgs, -1)
                test_dataset.imgs = np.expand_dims(test_dataset.imgs, -1)
                N_CHANNELS = 1
        
        dataset = (train_dataset.imgs, train_dataset.labels), (val_dataset.imgs, val_dataset.labels), (test_dataset.imgs, test_dataset.labels)
        train_ds, test_ds, validation_ds = brevis.dataset.prepare.dataset(dataset,
                                                                32,None,shuffle_size=15000,input_size=(input_size,input_size),
                                                                include_targets=False,num_outputs = N_CLASSES,reshuffle=True)
    
    meta_data = {'input_size': input_size, 'n_channels': N_CHANNELS, 'n_classes' : N_CLASSES}

    return train_ds, test_ds, validation_ds, meta_data