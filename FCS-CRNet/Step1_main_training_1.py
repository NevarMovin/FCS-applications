import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, subtract, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as sio 
import numpy as np
import random
import math
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import hdf5storage
import os

# settings of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.Session(config=config)

envir = 'indoor'  # 'indoor' as scenario_1

# main training params
epochs = 1000
batch_size = 200
dropout = 0.3
size_of_trainingset = 100000  # dataset size

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height*img_width*img_channels

# network params
encoded_dim = 512  # compress rate=1/4->dim.=512, 1/8->dim.=256, 1/16->dim.=128, 1/32->dim.=64


# Build the autoencoder model of CRNet
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def encoder_network(x):    
    # encoder
    sidelink = x

    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (1, 9), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (9, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    sidelink = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(sidelink)
    sidelink = add_common_layers(sidelink)

    x = concatenate([x, sidelink], axis=1)

    x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear')(x)
    encoded = Dropout(dropout)(encoded)  # dropout

    return encoded


def decoder_network(encoded):
    # decoder
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,))(x)

    x = Conv2D(2, (5, 5), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    # CRBlock
    for i in range(2):
        sidelink = x
        shortcut = x
        
        x = Conv2D(7, (3, 3), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (1, 9), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (9, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)

        sidelink = Conv2D(7, (1, 5), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)
        sidelink = Conv2D(7, (5, 1), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)

        x = concatenate([x, sidelink], axis=1)

        x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        
        x = add([x, shortcut])

    x = Activation('sigmoid')(x)
    return x


image_tensor = keras.Input(shape=(img_channels, img_height, img_width))
codewords_vector = keras.Input(shape=(encoded_dim,))

encoder = keras.Model(inputs=[image_tensor], outputs=[encoder_network(image_tensor)])
decoder = keras.Model(inputs=[codewords_vector], outputs=[decoder_network(codewords_vector)])
autoencoder = keras.Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT']

elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htrainout.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT']


x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train[0:size_of_trainingset]

# adapt this if using 'channels_first' image data format
x_train = np.reshape(x_train, [len(x_train), img_channels, img_height, img_width])
x_val = np.reshape(x_val, [len(x_val), img_channels, img_height, img_width])
x_test = np.reshape(x_test, [len(x_test), img_channels, img_height, img_width])


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


file = 'CRNet_' + envir + '_dim' + str(encoded_dim) + 'Step1_1'
file_encoder = 'CRNet_encoder_' + envir + '_dim' + str(encoded_dim) + 'Step1_1'
file_decoder = 'CRNet_decoder_' + envir + '_dim' + str(encoded_dim) + 'Step1_1'

path = 'result/TensorBoard_%s' % file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

history = LossHistory()

callbacks = [history, tf.keras.callbacks.TensorBoard(log_dir=path)]

autoencoder.fit(x=x_train, y=x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)


# store the parameters of the model
outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

outfile_encoder = 'result/%s_model.h5' % file_encoder
encoder.save_weights(outfile_encoder)

outfile_decoder = 'result/%s_model.h5' % file_decoder
decoder.save_weights(outfile_decoder)

print("Step1_main_training_1 has finished.")
