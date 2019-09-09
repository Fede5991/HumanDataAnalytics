from keras.models import Model, load_model

from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, LSTM, Bidirectional
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import optimizers
from keras_radam import RAdam

import keras
from keras import backend as K
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Conv2D,Conv1D,MaxPooling2D,AveragePooling1D,MaxPooling1D
from keras.layers import Dense,Flatten,Dropout
from keras import initializers,optimizers,backend as k
from keras_radam import RAdam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import confusion_matrix,accuracy_score
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def CNNLSTMAttention(network_input_shape, number_of_classes, optimizer, rnn_func = LSTM):

    inputs = Input(shape=(int(network_input_shape[0]), int(network_input_shape[1]), 1))
    
    x = Permute((2,1,3)) (inputs)
    
    x = Conv2D(10, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    x = Bidirectional(rnn_func(64, return_sequences = True, activation='tanh',recurrent_activation='sigmoid')) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(rnn_func(64, return_sequences = True, activation='tanh',recurrent_activation='sigmoid')) (x) # [b_s, seq_len, vec_dim]

    xFirst = Lambda(lambda q: q[:,49]) (x) #[b_s, vec_dim] #32
    query = Dense(128) (xFirst)

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]

    x = Dense(64, activation = 'relu')(attVector)
    x = Dense(32)(x)

    output = Dense(number_of_classes, activation = 'softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])
    
    # Optimizer choice
    if optimizer=='radam':
        Optimizer=RAdam()
    elif optimizer=='sgd':
        Optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif optimizer=='rmsprop':
        Optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer=='adagrad':
        Optimizer=keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    elif optimizer=='adadelta':
        Optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95,epsilon=None, decay=0.0)
    elif optimizer=='adam':
        Optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer=='adamax':
        Optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    else:
        Optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        
    model.compile(optimizer=Optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model