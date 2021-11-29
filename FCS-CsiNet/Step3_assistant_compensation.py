import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras import optimizers
import scipy.io as sio
import numpy as np
import math
import time
import hdf5storage
import os

# settings of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.Session(config=config)

envir = 'outdoor'  # 'outdoor' as scenario_2

# main training params
epochs = 1000
batch_size = 200
size_of_trainingset = 10000  # dataset size

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

# network params
residual_num = 2
encoded_dim = 512  # compress rate=1/4->dim.=512, 1/8->dim.=256, 1/16->dim.=128, 1/32->dim.=64


# Build the autoencoder model of CsiNet
def encoder_network(x):
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    # encoder
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear', name='encoded_layer')(x)

    return encoded


def decoder_network(encoded):
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    # decoder
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,), name='reconstructed_image')(x)
    for i in range(residual_num):
        x = residual_block_decoded(x)
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
input_vector = Input(shape=(encoded_dim,))

encoder = Model(image_tensor, encoder_network(image_tensor))
decoder = Model(input_vector, decoder_network(input_vector))

adam = optimizers.Adam(lr=0.0002)  # reset the learning rate
encoder.compile(optimizer=adam, loss='mse')
decoder.compile(optimizer=adam, loss='mse')

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
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


file = 'CsiNet_' + envir + '_dim' + str(encoded_dim) + 'Step3'

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
                             save_best_only=False)

history = LossHistory()

callbacks = [checkpoint, history, TensorBoard(log_dir=path)]

encoder.load_weights('result/CsiNet_encoder_outdoor_dim512_Step1_12_model.h5')

encoder.layers[5].trainable = False  # set the FC layer as untrainable

decoder.load_weights('result/CsiNet_decoder_outdoor_dim512_Step1_12_model.h5')

autoencoder = Model(image_tensor, decoder(encoder(image_tensor)))
autoencoder.compile(optimizer=adam, loss='mse')
print(autoencoder.summary())


autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)


# store the parameters of the model
outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)


# Testing data
outfile = 'result/%s_model.h5' % file
autoencoder.load_weights(outfile)

tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)

print("x_test_C shape is ", x_test_C.shape)
power = np.sum(abs(x_test_C) ** 2, axis=1)

mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("When dimension is", encoded_dim)
print("MSE is ", 10 * math.log10(np.mean(mse)))
print("NMSE is ", 10 * math.log10(np.mean(mse / power)))

print("Step3_assistant_compensation has finished.")
