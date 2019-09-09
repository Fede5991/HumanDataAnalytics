# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:27:25 2019

@author: Fede

"""

import numpy as np
from tqdm import tqdm
from plots import plot_AE_pre
import keras
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers import Conv2D,Conv1D,MaxPooling2D,AveragePooling1D,MaxPooling1D
from keras.layers import Dense,Flatten,Dropout,BatchNormalization,UpSampling2D
from keras import initializers,optimizers,backend as k

def pretraining1_AE(epochs,layers,filters,kernels,dropout,encoder,decoder,
                   input_img,training_set2,validation_set2):
    train_loss = []
    val_loss = []
    for layer in tqdm(layers):
        autoencoder = Model(input_img, decoder(encoder(input_img,layer,
                                                       filters[1],kernels[1],dropout[0]),
                                               layer,filters[1],kernels[1],dropout[0]))
        autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
        autoencoder_train = autoencoder.fit(training_set2, training_set2, batch_size=64,
                                            epochs=epochs,verbose=0,
                                            validation_data=(validation_set2, validation_set2))
        train_loss.append(autoencoder_train.history['loss'])
        val_loss.append(autoencoder_train.history['val_loss'])
    
    Epochs = np.linspace(1,epochs,epochs)
    plot_AE_pre(Epochs,train_loss,val_loss,layers,'#layers')
    lasts = []
    for i in range(3):
        lasts.append(val_loss[i][epochs-1])
    layer = layers[np.argmin(lasts)]
    
    train_loss = []
    val_loss = []
    for Filter in tqdm(filters):
        autoencoder = Model(input_img, decoder(encoder(input_img,layer,Filter,kernels[1],dropout[0]),
                                               layer,Filter,kernels[1],dropout[0]))
        autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
        autoencoder_train = autoencoder.fit(training_set2, training_set2, batch_size=64,
                                            epochs=epochs,verbose=0,
                                            validation_data=(validation_set2, validation_set2))
        train_loss.append(autoencoder_train.history['loss'])
        val_loss.append(autoencoder_train.history['val_loss'])
        
    plot_AE_pre(Epochs,train_loss,val_loss,filters,'#filters')
    lasts = []
    for i in range(3):
        lasts.append(val_loss[i][epochs-1])
    Filter = filters[np.argmin(lasts)]
    
    train_loss = []
    val_loss = []
    for kernel in tqdm(kernels):
        autoencoder = Model(input_img, decoder(encoder(input_img,layer,Filter,kernel,dropout[0]),
                                               layer,Filter,kernel,dropout[0]))
        autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
        autoencoder_train = autoencoder.fit(training_set2, training_set2, batch_size=64,
                                            epochs=epochs,verbose=0,
                                            validation_data=(validation_set2, validation_set2))
        train_loss.append(autoencoder_train.history['loss'])
        val_loss.append(autoencoder_train.history['val_loss'])
        
    plot_AE_pre(Epochs,train_loss,val_loss,kernels,'kernelsize')
    lasts = []
    for i in range(3):
        lasts.append(val_loss[i][epochs-1])
    kernel = kernels[np.argmin(lasts)]
    
    train_loss = []
    val_loss = []
    for drop in tqdm(dropout):
        autoencoder = Model(input_img, decoder(encoder(input_img,layer,Filter,kernel,drop),
                                               layer,Filter,kernel,drop))
        autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
        autoencoder_train = autoencoder.fit(training_set2, training_set2, batch_size=64,
                                            epochs=epochs,verbose=0,
                                            validation_data=(validation_set2, validation_set2))
        train_loss.append(autoencoder_train.history['loss'])
        val_loss.append(autoencoder_train.history['val_loss'])
        
    plot_AE_pre(Epochs,train_loss,val_loss,dropout,'dropout')
    lasts = []
    for i in range(3):
        lasts.append(val_loss[i][epochs-1])
    Dropout = dropout[np.argmin(lasts)]
    return layer,Filter,kernel,Dropout
    