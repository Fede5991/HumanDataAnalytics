# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:53:18 2019

@author: Fede
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix,accuracy_score

def plot_IAHOS(y, ogp, ogp2, tgp, tgp2, model):

    sp_titles = ("Mean train. accuracy first round",
                 "Mean valid accuracy first round",
                 "Mean train accuracy last round",
                 "Mean valid accuracy last round")

    n_rows = 2
    n_cols = 2

    fig = make_subplots(rows = n_rows, cols = n_cols, subplot_titles=sp_titles)

    x = np.linspace(0, len(tgp[0]) - 1, len(tgp[0]))
    Colorscale = [[0, '#FF0000'],[0.5, '#F1C40F'], [1, '#00FF00']]
    fig.add_trace(go.Heatmap(y = [y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp2, colorscale = Colorscale),row=1,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp,colorscale=Colorscale),row=1,col=2)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp2, colorscale = Colorscale),row=2,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp,colorscale=Colorscale),row=2,col=2)
    fig.update_layout(height=600, width=800)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/IAHOS_'+str(model)+'.png')
    fig.show()

def plot_confusion_matrix(new_test_labels, y_pred, words_name, feature,optimizer, root):
    title = 'Normalized confusion matrix'

    cm = confusion_matrix(y_true=new_test_labels,y_pred=y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize = (10,8), facecolor=(1, 1, 1))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.jet)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=words_name, yticklabels=words_name,
           title=title,
           ylabel='True word',
           xlabel='Predicted word')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(root + '/confusion_matrix_' + str(feature) + '_' + optimizer + '.png')
    plt.clf()


'''
def plot_confusion_matrix(new_test_labels, y_pred, words_name, model, root):
    cm = confusion_matrix(y_true=new_test_labels,y_pred=y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.jet)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(len(words_name))
    plt.xticks(tick_marks,words_name,rotation=90)
    plt.yticks(tick_marks,words_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(root + '/confusion_matrix_'+str(model)+'.png')
    plt.show()
'''
def plot_training_accuracy(training_accuracy,optimizers,model):
    plt.figure(figsize=(12,4))
    for i in range(len(optimizers)):
        plt.plot(training_accuracy[i])
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(optimizers)
    plt.ylim(0.7, 1)
    plt.grid(True)
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/Training_accuracy'+str(model))
    plt.show()

def plot_validation_accuracy(validation_accuracy,optimizers,model):
    plt.figure(figsize=(12,4))
    for i in range(len(optimizers)):
        plt.plot(validation_accuracy[i])
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(optimizers)
    plt.ylim(0.7, 1)
    plt.grid(True)
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/Validation_accuracy'+str(model))
    plt.show()

def plot_test_scores(scores,y,model):
    fig = go.Figure(data=[go.Bar(name='radam', x=scores, y=y[0],text=y[0]),
                      go.Bar(name='sgd', x=scores, y=y[1],text=y[1]),
                      go.Bar(name='rmsprop', x=scores, y=y[2],text=y[2]),
                      go.Bar(name='adagrad',x=scores,y=y[3],text=y[3]),
                      go.Bar(name='adadelta', x=scores, y=y[4],text=y[4]),
                      go.Bar(name='adam', x=scores, y=y[5],text=y[5]),
                      go.Bar(name='adamax', x=scores, y=y[6],text=y[6]),
                      go.Bar(name='nadam', x=scores, y=y[7],text=y[7])])
    fig.update_layout(barmode='group',width=800)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/test_scores_'+str(model)+'.png')
    fig.show()

def plot_output_NN(words_name,classifier,audio_signal):
    plt.figure(figsize=(14,4))
    plt.bar(words_name,classifier.predict(audio_signal)[0])
    plt.title('Probability distribution per class')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=90)
    plt.show()

def plot_AE_pre(Epochs,train_loss,val_loss,params,name_param):
    last=int(Epochs[-1])
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    plt.setp(ax.flat, xlabel='Epochs', ylabel='Accuracy')
    ax[0].set_title('Training accuracy w.r.t.'+name_param)
    for i in range(3):
        ax[0].plot(Epochs, train_loss[i],label=params[i])
    ax[0].legend()

    ax[1].set_title('Validation accuracy w.r.t.'+name_param)
    for i in range(3):
        plt.plot(Epochs, val_loss[i],label=params[i])
    ax[1].legend()
    plt.savefig('Train and val accuracy wrt'+name_param)
    plt.show()

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    plt.setp(ax.flat, xlabel='last epochs', ylabel='Accuracy')
    ax[0].set_title('Training accuracy w.r.t. '+name_param+' last epochs')
    for i in range(3):
        ax[0].plot(Epochs[last-10:], train_loss[i][last-10:],label=params[i])
    ax[0].legend()

    ax[1].set_title('Val accuracy w.r.t. '+name_param+' last epochs')
    for i in range(3):
        plt.plot(Epochs[last-10:], val_loss[i][last-10:],label=params[i])
    ax[1].legend()
    plt.savefig('Train and val accuracy last epochs wrt '+name_param)
    plt.show()
