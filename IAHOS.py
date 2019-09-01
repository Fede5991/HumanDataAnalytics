# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:37:22 2019

@author: Fede
"""
import numpy as np
from hyperparams_initialization import hyperparams_initialization
from extraction_performances import extraction_performances
from tqdm import tqdm
import keras
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers import Conv3D,MaxPooling3D,Dense,Flatten,Dropout
from keras import initializers,optimizers,backend as k
from time import time

def IAHOS(rounds,method,limits,attempts,variables,iterations,Classifier,
          training_set,training_labels,validation_set,validation_labels):
    for r in range(rounds):
        print ("Round ",r+1," of ",rounds)
        epochs=1
        if r>0:
            attempts=2
        hm,p_values = hyperparams_initialization(attempts,variables,method,limits,r)
    
        training_accuracy = []
        validation_accuracy = []
        for i in tqdm(range(iterations)):
            classifier=Classifier(32,i,hm,'adam')
            history=classifier.fit(training_set,training_labels,
                                   validation_data=[validation_set,validation_labels],
                                   epochs=epochs,batch_size=40,verbose=0)
    
            training_accuracy.append(history.history['acc'][-1])
            validation_accuracy.append(history.history['val_acc'][-1])
            del classifier
            del history
            K.clear_session()
        general_perf,general_perf2=extraction_performances(validation_accuracy,training_accuracy,variables,iterations,attempts)
        del training_accuracy
        del validation_accuracy
        
        if r==0:
            old_general_perf = general_perf
            old_general_perf2 = general_perf2
            temp_general_perf=general_perf
            temp_general_perf2 = general_perf2
        else:
            new_p_values = []
            for s in range(variables):
                a = []
                a.append(p_values[s])
                b = []
                b.append(temp_p_values[s])
                c=np.concatenate((a,b),axis=1)
                new_p_values.append(np.sort(c)[0])
            p_values = new_p_values
            new_values = []
            new_values2 = []
            for t in range(len(temp_general_perf)):
                new_values.append(np.insert(temp_general_perf[t],indeces[t][1],general_perf[t]))
                new_values2.append(np.insert(temp_general_perf2[t],indeces[t][1],general_perf2[t]))
            temp_general_perf=new_values
            temp_general_perf2=new_values2
        
        indeces = []
        best_hp=[]
        final = []
        for i in range(variables):
            values = []
            index = []
            index.append(np.argsort(temp_general_perf[i])[-1])
            values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]])
            final.append(p_values[i][np.argsort(temp_general_perf[i])[-1]])
            if np.argsort(temp_general_perf[i])[-1]>0:
                values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]-1])
                index.append(np.argsort(temp_general_perf[i])[-1]-1)
            else:
                values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]+1])
                index.append(np.argsort(temp_general_perf[i])[-1]+1)
            best_hp.append(np.sort(values))
            indeces.append(np.sort(index))
        limits = best_hp
        temp_p_values = p_values
    return temp_general_perf,temp_general_perf2,old_general_perf,old_general_perf2,final