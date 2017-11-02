# -*- coding: utf-8 -*-

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
import math
import pickle

from keras import backend as K
K.clear_session()

#import pylab as plt
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
map_height, map_width = 16, 8  # grid size
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
days_test = 10
T = 24
len_test = T * days_test
lr = 0.001  # learning rate
nb_epoch =40
nb_cont_epoch = 10
batch_size =32
seq_length=12
PATH_RESULT='Test_RET'
PATH_MODEL='Test_MODEL'
num_layers=3
test_data_nums =None

def build_model():
    seq = Sequential()
    
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       input_shape = (None, 2, map_height, map_width),
                       padding='same', activation='relu', return_sequences=True, data_format='channels_first'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
         padding='same', activation='relu', return_sequences=True, data_format='channels_first'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
         padding='same', activation='tanh', return_sequences=False, data_format='channels_first'))
    seq.add(BatchNormalization())

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
    X_train, Y_train, X_test, Y_test, X_timestamps, Y_timestamps, mmn = BikeNYC.load_sequence(seq_length=seq_length, T=24, 
                                        test_percent=0.055, data_numbers=test_data_nums)   
    print('X_train shape is', X_train.shape)
    print('Y_train shape is', Y_train.shape)
    print('X_test shape is', X_test.shape)
    print('Y_test shape is', Y_test.shape)
    
    # Train the network, to use reset_states(), 
    # we must make epochs=1 and train for nb_epoch in a Loop.
    seq = build_model()

    hyperparams_name = 'b{}.Conv2DLSTM_layers{}.SeqLen{}.Conv2D_LSTM_BikeNYC.lr{}'.format(
        batch_size, num_layers, seq_length, lr)
    fname_param = os.path.join(PATH_MODEL, '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_rmse', patience=10, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    # for e in range(nb_epoch):
    #     seq.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
    #     seq.reset_states()

    history = seq.fit(X_train, Y_train, 
        batch_size=batch_size, 
        epochs=nb_epoch, 
        validation_split=0.1,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1)

    seq.save_weights(os.path.join(
        PATH_MODEL, '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        PATH_RESULT, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')

    seq.load_weights(fname_param)
    score = seq.evaluate(X_train, Y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    score = seq.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        PATH_MODEL, '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')

    history = seq.fit(X_train, Y_train, 
        nb_epoch=nb_cont_epoch, 
        verbose=1, 
        batch_size=batch_size, 
        callbacks=[model_checkpoint], 
        validation_data=(X_test, Y_test))

    pickle.dump((history.history), open(os.path.join(
        PATH_RESULT, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    seq.save_weights(os.path.join(
        PATH_MODEL, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = seq.evaluate(X_train, Y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    score = seq.evaluate(
        X_test, Y_test, batch_size=batch_size, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

if __name__ == '__main__':
    main()
