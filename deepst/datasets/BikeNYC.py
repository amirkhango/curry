# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle as pickle
import numpy as np

from . import load_stdata
from ..preprocessing import MinMaxNormalization
from ..preprocessing import remove_incomplete_days
from ..config import Config

from ..datasets.STMatrix import STMatrix

from ..preprocessing import timestamp2vec
np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH

# def get_batch(dataseq, seq_length=10, batch_size=1):
    
#     assert len(dataseq) > seq_length, 'Error! Length of data sequences is <= seq_length'
  
#     if len(dataseq)%seq_length == 0:
#         dataseq = dataseq[:-seq_length+1]
#     else:
#         dataseq = dataseq[:int(len(dataseq)//seq_length*seq_length)+1]

#     X_data = np.copy(dataseq[:-1])
#     Y_data = np.copy(dataseq[seq_length::seq_length])
#     print('X_data shape is', X_data.shape)
#     X_data = np.array(np.split(X_data, len(X_data)//seq_length ))

#     print('X_data shape is', X_data.shape)
#     print('Y_data shape is', Y_data.shape)
    


#     return X_data, Y_data



def load_sequence(seq_length=5, T=24, test_percent=0.1, data_numbers=None):
    # load data
    dataseq, timestamps = load_stdata(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'),data_numbers=data_numbers)
    # remove a certain day which does not have 24 (for TaxiBJ data is 48) timestamps
    dataseq, timestamps = remove_incomplete_days(dataseq, timestamps, T)
    print('The length of dataseq is:', len(dataseq))
    
    assert (dataseq >= 0).all() , 'There are error data which are < 0'    
    assert len(dataseq) > seq_length, 'Error! Length of data sequences is <= seq_length'

    X_data = np.copy(dataseq[:-1])
    X_timestamps = np.copy(timestamps[:-1])
    Y_data = np.copy(dataseq[seq_length:])
    Y_timestamps = np.copy(timestamps[seq_length:])

    assert len(X_data)==(len(Y_data)-1+seq_length), 'Error! Length(input) != Length(label)+seq_length'

    len_test = int(test_percent*len(X_data))

    #Normalization
    #print(X_data[:-len_test].shape, Y_data[:-len_test][-1].shape)
    train_data=np.append(X_data[:-len_test],[Y_data[:-len_test][-1]], axis=0)
    mmn=MinMaxNormalization()
    mmn.fit(train_data)
    X_data=mmn.transform(X_data)
    Y_data=mmn.transform(Y_data)

    #Prepare input sequence and label for training and testing
    X_list=[]
    X_timestamps_list=[]

    for i in range( len(X_data)-(seq_length-1) ):
        X_list.append(X_data[i:i+seq_length])
        X_timestamps_list.append(X_timestamps[i:i+seq_length])

    X_data = np.array(X_list)
    X_timestamps = np.array(X_timestamps_list)
    print('X_data shape is', X_data.shape)
    print('Y_data shape is', Y_data.shape)
    

    #Y_data = np.expand_dims(Y_data, axis=1)
    #X_data = np.array(np.split(X_data, len(X_data)//seq_length ))

    X_train = X_data[:-len_test]
    Y_train = Y_data[:-len_test]
    X_test = X_data[-len_test:]
    Y_test =  Y_data[-len_test:]
   
    #train_data = np.concatenate(X_train,Y_train)
    #print('train_data shape:..', train_data.shape)

    return X_train, Y_train, X_test, Y_test, X_timestamps, Y_timestamps, mmn



def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, 
    preprocess_name='preprocessing.pkl', meta_data=True, data_numbers=None):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'),data_numbers=data_numbers)
    print('h5 data shape: {}'.format(data.shape))
    print('h5 timestamps data shape: {}'.format(timestamps.shape))
    print('The first 3 timestamps are: {}'.format(timestamps[:3]))
    print('The Last 3 timestamps are: {}'.format(timestamps[-3:]))
    # print(timestamps)da
    # remove a certain day which does not have 24 (for TaxiBJ data is 48) timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    print('sequences of all data shape is: {}'.format(data.shape))
    assert (data >= 0).all() , 'There are error data which are < 0' 
    #data[data < 0] = 0.
    data_all = [data]
    #print('data_all shape is: {}',(data_all.shape)) #AttributeError: 'list' object has no attribute 'shape'
    timestamps_all = [timestamps]
    # minmax_scale
    print('len_test is', len_test)
    data_train = data[:-len_test]
    print('sequences of data_train  shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    print('length of data_all is',len(data_all))
    for d in data_all:
        print('d len is',d.shape)
        data_all_mmn.append(mmn.transform(d))

    fpkl = open('preprocessing.pkl', 'wb')
    print('[mmn] length is', len([mmn]))
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XALL, XC, XP, XT =[], [], [], []
    Y = []
    timestamps_Y = []
    print('length of data_all_mmn is',len(data_all_mmn))
    print('length of timestamps_all is',len(timestamps_all))
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)

    X_ALL = np.concatenate((XC,XP,XT), axis=1)
    print('X_ALL shape is:', X_ALL.shape)
    #X_train_ALL = X_ALL[:-len_test]
    #X_test_ALL = X_ALL[-len_test:]
    #print('X_train_ALL shape is:', X_train_ALL.shape)
    #print('X_test_ALL shape is:', X_test_ALL.shape)

    Y = np.vstack(Y)
    print('len_test is',len_test)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []

    X_train_ALL =[]
    X_test_ALL = []
    X_train_ALL.append(X_ALL[:-len_test])
    X_test_ALL.append(X_ALL[-len_test:])

    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        print('l is',l)
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        print('meta_feature shape is:',meta_feature.shape)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
        X_train_ALL.append(meta_feature_train)
        X_test_ALL.append(meta_feature_test)

    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train_ALL, X_test_ALL, X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

if __name__ == '__main__':
    load_sequence(seq_length=5, T=24, test_percent=0.1, data_numbers=None)