'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    Add
)
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
#from keras.utils.visualize_util import plot
import numpy as np

def _shortcut(input, residual):
    return Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(filters=nb_filter, kernel_size=(3,3), strides=subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f

def _conv_unit(nb_filter, kernel_size=(3,3), strides=(1,1), padding="same"):
    #activation = Activation('relu')(input)
    def  f(input):
        input = Conv2D(filters=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        activation = Activation('relu')(input)
        return activation
    return f

def DeepConvNets(conv_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = conv_unit(nb_filter,strides=(1,1))(input) 
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    main_outputs = []
    input_channels = 0

    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input_channels += len_seq

    input = Input(shape=(nb_flow * input_channels, map_height, map_width))
    main_inputs.append(input)           
    output = DeepConvNets(_conv_unit, nb_filter=64, 
                        repetations= 5)(input)

    #output = Activation('tanh')(output)
    main_outputs.append(output)        
    model = Model(outputs=main_outputs, inputs=main_inputs)

    # print('outputs length:',len(outputs))
    # # parameter-matrix-based fusion
    # if len(outputs) == 1:
    #     main_output = outputs[0]
    # else:
    #     from .iLayer import iLayer
    #     new_outputs = []
    #     for output in outputs:
    #         new_outputs.append(iLayer()(output))
    #     #print('new_outputs shape is:',np.asarray(new_outputs).shape)
    #     main_output = Add()(new_outputs)

    # # fusing with external component
    # if external_dim != None and external_dim > 0:
    #     # external input
    #     external_input = Input(shape=(external_dim,))
    #     main_inputs.append(external_input)
    #     embedding = Dense(output_dim=10)(external_input)
    #     embedding = Activation('relu')(embedding)
    #     h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
    #     activation = Activation('relu')(h1)
    #     external_output = Reshape((nb_flow, map_height, map_width))(activation)
    #     main_output = Add()([main_output, external_output])
    # else:
    #     print('external_dim:', external_dim)

    #main_output = Activation('tanh')(main_output)
    #model = Model(input=main_inputs, output=main_output)

    return model

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
