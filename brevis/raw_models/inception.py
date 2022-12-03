import os 
import sys
# os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+':'+os.environ['CONDA_PREFIX']+'/lib/'
# sys.path.append('..')
from brevis import dataset
import tensorflow as tf

def get_model(INPUT_SIZE, N_CLASSES, N_CHANNELS, weights=None):
    base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(INPUT_SIZE, INPUT_SIZE, N_CHANNELS),
         weights=weights,include_top=False)


    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="classification")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    return model

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass    
    INPUT_SIZE = 224
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
    model.evaluate(test_ds, batch_size=32)
    model.save("inception_finetuned.hdf5")
