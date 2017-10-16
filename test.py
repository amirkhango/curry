from __future__ import print_function
import os
import pickle
import numpy as np
import math
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.models.STConvolution import binCNN
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from importlib import reload
from keras.utils.vis_utils import plot_model
from testmodel import stresnet as st

np.random.seed(1337)  # for reproducibility

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day

lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 da ys
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
print('factor: ', m_factor)
path_result = 'Test_RET'
path_model = 'Test_MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    #model = st(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
    #                 external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    
    model = binCNN(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf)
    
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()    
    return model

def main():
	model = build_model(external_dim=8)
	plot_model(model, to_file= os.path.join(path_model,'testmodel.png'), show_shapes=True)
	
	print("loading data...")
	three_models = True # If true, split Closeness, Period and Trend into 3 sub-CNN respectively.

	# data_numbers=None will use all data, this could be very slowly.
	# data_numbers=800 will use only 800 series for trying on small data.
	X_train_ALL, X_test_ALL, X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
	    T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
	    preprocess_name='preprocessing.pkl', meta_data=True, data_numbers=1500)

	print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

	print('=' * 10)
	print("compiling model...")
	print(
	    "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")
	print('external_dim is:', external_dim)

	hyperparams_name = 'binCNN_c{}.p{}.t{}.resunit{}.lr{}'.format(\
	    len_closeness, len_period, len_trend, nb_residual_unit, lr)
	fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))

	early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
	model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

	print('=' * 10)
	print("training model...")
	history = model.fit(X_train_ALL, Y_train,
	                    epochs=nb_epoch,
	                    batch_size=batch_size,
	                    validation_split=0.1,
	                    callbacks=[early_stopping, model_checkpoint],
	                    verbose=1)
	model.save_weights(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
	pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
	print('=' * 10)
	print('evaluating using the model that has the best loss on the valid set')

	model.load_weights(fname_param)
	score = model.evaluate(X_train_ALL, Y_train, batch_size=Y_train.shape[
	                       0] // 48, verbose=0)
	print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
	      (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

	score = model.evaluate(
	    X_test_ALL, Y_test, batch_size=Y_test.shape[0], verbose=0)
	print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
	      (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

	print('=' * 10)
	print("training model (cont)...")
	fname_param = os.path.join(
	    path_model, '{}.cont.best.h5'.format(hyperparams_name))
	model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
	history = model.fit(X_train_ALL, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
	                    model_checkpoint], validation_data=(X_test, Y_test))
	pickle.dump((history.history), open(os.path.join(
	    path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
	model.save_weights(os.path.join(
	    path_model, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

	print('=' * 10)
	print('evaluating using the final model')
	score = model.evaluate(X_train_ALL, Y_train, batch_size=Y_train.shape[
	                       0] // 48, verbose=0)
	print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
	      (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

	score = model.evaluate( 
	    X_test_ALL, Y_test, batch_size=Y_test.shape[0], verbose=0)
	print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
	      (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

if __name__ == '__main__':
    main()
