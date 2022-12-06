import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    help='datasete name')
parser.add_argument('model',
                    help='model name')
parser.add_argument('base_model',
                    help='base_model name')
parser.add_argument('num_branches', type=int,
                    help='num branches')
parser.add_argument('--out_dataset',
                    help='ood dataset name')    
parser.add_argument('--pretrained',
                    help='pretrained model')                
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
parser.add_argument('--gamma', type=float, default=0.0001,
                    help='gamma')
parser.add_argument('--lambda_t', type=float, default=60,
                    help='initial lambda')
parser.add_argument('--eval', action='store_true',
                    help='eval mode')
parser.add_argument('--first_thresholds',  nargs='+', type=float,
                    help='manual set the theshold of the first branch')
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

args = parser.parse_args()

model_dir = os.path.join(args.model_dir, '{}_{}'.format(args.dataset, args.model))
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

MODEL_PATH = os.path.join(model_dir, 
        '{}_{}_{}_{}branches_lambda{}_gamma{}_seed{}.hdf5'.format(
        args.dataset, args.model, args.base_model, args.num_branches, args.lambda_t, args.gamma, args.seed))



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

model_to_class = {'alexnet': raw_models.alexnet.get_model,
                 'resnet': raw_models.resnet.get_model,
                 'inception': raw_models.inception.get_model,
                 'wideresnet': raw_models.wide_resnet_pretrain.get_model}
model_to_attach_pts = {'alexnet': ['max_pooling2d', 'max_pooling2d_1', 'conv2d_3'],
                 'resnet': ["conv2_block1_out","conv2_block3_out", 'conv3_block1_out'],
                 'inception': ["mixed0","mixed1", 'mixed2'],
                 'wideresnet': ["conv2_block1_out","conv2_block3_out", 'conv3_block1_out']}
branch_functs = [lambda *args, **kwargs: branch_conv2d(*args, depths = [64, 256], **kwargs),
                lambda *args, **kwargs: branch_conv2d(*args, depths = [128, 512], **kwargs),
                lambda *args, **kwargs: branch_conv2d(*args, depths = [128, 512], **kwargs)]

if args.model == 'branchy' or args.eval:
    model = model_to_class[args.base_model](meta_data['input_size'], meta_data['n_classes'], meta_data['n_channels'], 'imagenet' if args.dataset == 'cifar10' else None)
    model = brevis.BranchModel(model=model, custom_objects={})
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,  beta_1=0.99,  beta_2=0.999,)

    branch_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    trunk_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    if args.model == 'brevis':
        metrics = ["energy"]
    else:
        metrics = ["entropy", "uncert", "calibration"]

elif args.model == 'brevis':
    model = model_to_class[args.base_model](meta_data['input_size'], meta_data['n_classes'], meta_data['n_channels'], 'imagenet' if args.dataset == 'cifar10' else None)
    model = brevis.BranchModel(model=model, custom_objects={})
    model.load_weights(args.pretrained)
    optimizer = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    growth_callback = growth_update(annealing_rate = 60, start_t = 0.,max_t = args.lambda_t, starting_epoch =1, branch_names=["branch_exit_accuracy","branch_exit_1_accuracy"])
    branch_loss = Brevis_loss_final(growth_callback, args.gamma)  ### the brevisnet loss function.
    trunk_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = ["energy"]

else:
    raise NotImplementedError

model = brevis.branches.branch.add_branches(model, branch_functs[:args.num_branches],
                                           branchPoints = model_to_attach_pts[args.base_model][:args.num_branches],
                                            target_input=False,loop=False,num_outputs=meta_data['n_classes'])
model.setFrozenTraining(args.model == 'brevis')

if not args.eval:
    model.compile(loss=[trunk_loss,branch_loss,branch_loss], 
                    optimizer=optimizer,
                    metrics=['accuracy'])

    earlystop1 = tf.keras.callbacks.EarlyStopping(monitor='val_branch_exit_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto',restore_best_weights=True)
    earlystop2 = tf.keras.callbacks.EarlyStopping(monitor='val_branch_exit_1_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto',restore_best_weights=True)
    earlystop3 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    model.fit(train_ds, epochs=args.epochs, validation_data = validation_ds, batch_size=args.batch_size,callbacks=[earlystop1, earlystop2, earlystop3])
    model.save(MODEL_PATH)
else:
    model.load_weights(args.pretrained)
print(model.summary())
try:
    flops = get_branched_flops(model, ["branch_exit"] + ["branch_exit_{}".format(i + 1) for i in range(args.num_branches-1)] + ["classification"])
except:
    flops = get_branched_flops(model, ["branch_exit"] + ["branch_exit_{}".format(i + 1) for i in range(args.num_branches-1)] + ["dense_2"])

print(f"FLOPS: ", flops)
output_branchy_ID= getPredictions_Energy(model, test_ds,stopping_point=None)
if args.out_dataset:
    _, test_ds_ood, _, _ = load_dataset(args.out_dataset, 224 if args.dataset == 'cifar10' else 96)
    output_branchy_OOD= getPredictions_Energy(model, test_ds_ood,stopping_point=None)[:len(output_branchy_ID)]
    if args.first_thresholds:
        for first_thresh in args.first_thresholds:
            for metric in metrics:
                infer_result_OOD(output_branchy_ID, output_branchy_OOD, [metric], threshold='gmean', flops=flops, first_thresh=first_thresh)
                print("="*100)
        print("=-"*50)
    else:
        for metric in metrics:
            infer_result_OOD(output_branchy_ID, output_branchy_OOD, [metric], threshold='gmean', flops=flops)
            print("="*100)
else:
    for metric in metrics:
        infer_result(output_branchy_ID, [metric], threshold='gmean', flops=flops)
        print("="*100)
