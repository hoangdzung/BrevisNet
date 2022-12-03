# import os 
import sys
# os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+':'+os.environ['CONDA_PREFIX']+'/lib/'
# sys.path.append('..')
from brevis import dataset
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.optimizers import SGD

def get_model(INPUT_SIZE, N_CLASSES=10, N_CHANNELS=3, **kwargs):
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(INPUT_SIZE,INPUT_SIZE,N_CHANNELS)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    
    return model

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    INPUT_SIZE = 32
    N_CLASSES = 10
    N_CHANNELS = 3
    train_ds, test_ds, validation_ds = dataset.prepare.dataset(tf.keras.datasets.cifar10.load_data(),
                                                                    32,5000,shuffle_size=15000,input_size=(INPUT_SIZE,INPUT_SIZE),
                                                                    include_targets=False,num_outputs = N_CLASSES,reshuffle=True)
    model = get_model(INPUT_SIZE, N_CLASSES, N_CHANNELS)
    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
            )

    model.fit(train_ds,
        epochs=50,
        validation_data = validation_ds,
        validation_freq = 1,
        callbacks=[earlystop],
        batch_size=32
        )

    test_scores = model.evaluate(test_ds)

    print("overall loss: {}".format(test_scores[0]))
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save('alexNet.hdf5')
    print("Task Complete")

    