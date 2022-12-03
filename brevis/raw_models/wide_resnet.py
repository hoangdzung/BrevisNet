import os
import sys
# os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+':'+os.environ['CONDA_PREFIX']+'/lib/'
sys.path.append('..')# 
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

INPUT_SIZE = 32
train_ds, test_ds, validation_ds = dataset.prepare.dataset(tf.keras.datasets.cifar10.load_data(),
                                                                32,5000,shuffle_size=15000,input_size=(INPUT_SIZE,INPUT_SIZE),
                                                                include_targets=False,num_outputs = 10,reshuffle=True)

nb_classes = 10
dropout_probability = 0 # table 6 on page 10 indicates best value (4.17) CIFAR-10
weight_decay = 0.0005   # page 10: "Used in all experiments"
lr_schedule = [60, 120, 160] # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008

# Other config from code; throughtout all layer:
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua
channel_axis = 1


# ================================================

# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]
        
        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=channel_axis)(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(n_bottleneck_plane, 
                               (v[0],v[1]),
                                strides=v[2],
                                padding=v[3],
                                kernel_initializer=weight_init,
                                kernel_regularizer=l2(weight_decay),
                                use_bias=use_bias)(convs)
            else:
                convs = BatchNormalization(axis=channel_axis)(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability)(convs)
                convs = Conv2D(n_bottleneck_plane, 
                               (v[0],v[1]),
                                strides=v[2],
                                padding=v[3],
                                kernel_initializer=weight_init,
                                kernel_regularizer=l2(weight_decay),
                                use_bias=use_bias)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(n_output_plane, 
                              (1,1),
                              strides=stride,
                              padding="same",
                              kernel_initializer=weight_init,
                              kernel_regularizer=l2(weight_decay),
                              use_bias=use_bias)(net)
        else:
            shortcut = net

        return Add()([convs, shortcut])
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, stride=(1,1))(net)
        return net
    
    return f


def create_model(depth, input_shape, k):
    
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k]


    conv1 = Conv2D(n_stages[0], 
                    (3, 3), 
                    strides=1,
                    padding="same",
                    kernel_initializer=weight_init,
                    kernel_regularizer=l2(weight_decay),
                    use_bias=use_bias)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm)
                                            
    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
    flatten = Flatten()(pool)
    predictions = Dense(units=nb_classes, kernel_initializer=weight_init, use_bias=use_bias,
                        kernel_regularizer=l2(weight_decay))(flatten)

    model = Model(inputs=inputs, outputs=predictions)
    return model

model = create_model(depth=28, k=10, input_shape=(INPUT_SIZE,INPUT_SIZE,3))
model.summary()

model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])


model.fit(train_ds, epochs=200, validation_data = validation_ds, batch_size=128)
model.evaluate(test_ds, batch_size=32)
model.save("resnet50_finetuned.hdf5")