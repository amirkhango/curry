""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from keras import backend as K
K.clear_session()

#import pylab as plt
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
map_height, map_width = 16, 8  # grid size
days_test = 10
T = 24
len_test = T * days_test
lr = 0.0002  # learning rate

def build_model():
    seq = Sequential()
    
    seq.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                       input_shape = (None, 2, map_height, map_width),
                       padding='same', return_sequences=False, data_format='channels_first'))
    
    seq.add(Conv2D(filters=2, kernel_size=(3, 3),
                   activation='tanh',
                   padding='same', data_format='channels_first'))

    adam = Adam(lr=lr)
    seq.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    seq.summary()

    return seq

def main():
    # Load data
    print("loading data...")
    X_train, Y_train, X_test, Y_test, X_timestamps, Y_timestamps = BikeNYC.load_sequence(seq_length=3, T=24, 
                                        test_percent=0.1, data_numbers=None)   
    print('X_train shape is', X_train.shape)
    print('Y_train shape is', Y_train.shape)
    print('X_test shape is', X_test.shape)
    print('Y_test shape is', Y_test.shape)
        
    # Train the network
    seq = build_model()
    seq.fit(X_train, Y_train, batch_size=3,
            epochs=10, validation_split=0.1)
    os._exit()
    # Testing the network on one movie
    # feed it with the first 7 positions and then
    # predict the new positions
    which = 10
    track = noisy_movies[which][:2, ::, ::, ::]

    for j in range(2):
        new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
        print('new_pos shape is:....',new_pos.shape)
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)


    # And then compare the predictions
    # to the ground truth
    path_result = './Test_RET/'
    track2 = noisy_movies[which][::, ::, ::, ::]
    for i in range(3):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(121)

        if i >= 2:
            ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
        else:
            ax.text(1, 3, 'Initial trajectory', fontsize=20)

        toplot = track[i, ::, ::, 0]

        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth', fontsize=20)

        toplot = track2[i, ::, ::, 0]
        if i >= 2:
            toplot = shifted_movies[which][i - 1, ::, ::, 0]

        plt.imshow(toplot)
        plt.savefig((path_result+'%i_animate.png') % (i + 1))


if __name__ == '__main__':
    main()
