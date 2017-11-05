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
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
#from keras.utils.visualize_util import plot
import numpy as np

def threeway(c_conf=(3, 2, 32, 32), p_conf=(2, 2, 32, 32), t_conf=(2, 2, 32, 32), external_dim=None):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []

    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(None, nb_flow, map_height, map_width))
            main_inputs.append(input)

            # ConvLSTM1
            convlstm1 = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       input_shape = (None, 2, map_height, map_width),
                       padding='same', activation='relu', return_sequences=False, 
                       data_format='channels_first')(input)

            convlstm1 = BatchNormalization()(convlstm1)

            deconv = Conv2D(filters=2, kernel_size=(3, 3),
                   activation=None,
                   padding='same', data_format='channels_first')(convlstm1)

            outputs.append(deconv)

    print('outputs length:',len(outputs))
    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        #print('new_outputs shape is:',np.asarray(new_outputs).shape)
        main_output = Add()(new_outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = Add()([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model

if __name__ == '__main__':
    model = threeway()
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
