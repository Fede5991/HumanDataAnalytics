# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:43:07 2019

@author: Fede
"""
import numpy as np

def hyperparams_initialization(attempts,hp_number,method,limits,iteration):
    hm = np.zeros((hp_number,attempts**hp_number))
    hm_goodness = np.zeros((hp_number,attempts**hp_number))
    
    if (attempts>2 or iteration>0):
        possible_values = []
        for k in range(hp_number):
            values = []
            if method=='grid':
                if iteration==0:
                    values.append(limits[k][0])
                    for l in range(attempts-1):
                        values.append(limits[k][0]+(l+1)*(limits[k][-1]-limits[k][0])/(attempts-1))
                else:
                    for l in range(attempts):
                        values.append(limits[k][0]+(l+1)*(limits[k][-1]-limits[k][0])/(attempts+1)) 
            else:
                if iteration==0:
                    values.append(limits[k][0])
                    values.append(limits[k][-1])
                    values.append(np.sort(np.random.uniform(limits[k][0],limits[k][-1],size=(attempts-2))))
                else:
                    values.append(np.sort(np.random.uniform(limits[k][0],limits[k][-1],size=(attempts))))
            possible_values.append(np.sort(values))
    for i in range(attempts**hp_number):
        for j in range(hp_number):
            if attempts==1:
                hm[j,i]=(limits[j][0]+limits[j][-1])/2
            elif (attempts==2 and iteration==0):
                hm[j,i]=limits[j][int(i/(attempts**j))%attempts]
            else:
                hm[j,i]=possible_values[j][int(i/(attempts**j))%attempts]
    if (attempts>2 or iteration>0):
        return hm,possible_values
    else:
        return hm,limits

