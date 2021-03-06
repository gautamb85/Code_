
import htkmfc
import os
import lasagne
import time
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import ipdb

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

import h5py
from fuel.converters.base import fill_hdf5_file

def load_dataset():
    
    f1 = open('/Users/gautamB/dnn/Code_/spk_softmax/Train_feats_labs.plst')
    lines = f1.readlines()
    lines = [l.strip() for l in lines]
    labelz = [int(l.split()[1]) for l in lines] 
    #labelz = labelz[:20]
    features = [l.split()[0] for l in lines] 
    
    f2 = open('/Users/gautamB/dnn/Code_/spk_softmax/Valid_feats_labs.plst')
    lines = f2.readlines()
    lines = [l.strip() for l in lines]
    val_labelz = [int(l.split()[1]) for l in lines] 
    #val_labelz = val_labelz[:20]
    val_features = [l.split()[0] for l in lines] 
    
    n_samp = len(features)
    maxlen=600 #pad all utterances to this length
    feat_dim=20
    nSpk = 98
    dpth = '/Users/gautamB/0101_data/VQ_VAD_HO_EPD/'

    Data = np.zeros((n_samp, maxlen, feat_dim), dtype='float32')
    Mask = np.zeros((n_samp,maxlen), dtype='float32')
    #Targets = np.zeros((n_samp, nSpk), dtype='int32')
    
    vn_samp = len(val_features)
    val_Data = np.zeros((vn_samp, maxlen, feat_dim), dtype='float32')
    val_Mask = np.zeros((vn_samp,maxlen), dtype='float32')
    #Targets = np.zeros((n_samp, nSpk), dtype='int32')

    for ind,f in enumerate(features):
        fname = os.path.join(dpth,f+'.fea')
        fi = htkmfc.HTKFeat_read(fname)
        data = fi.getall()[:,:20]
        Mask[ind,:data.shape[0]] = 1.0
        pad = maxlen - data.shape[0]
        data = np.vstack((data, np.zeros((pad,20), dtype='float32')))
        Data[ind,:,:] = data
        

    for ind,f in enumerate(val_features):
        fname = os.path.join(dpth,f+'.fea')
        fi = htkmfc.HTKFeat_read(fname)
        data = fi.getall()[:,:20]
        val_Mask[ind,:data.shape[0]] = 1.0
        pad = maxlen - data.shape[0]
        data = np.vstack((data, np.zeros((pad,20), dtype='float32')))
        val_Data[ind,:,:] = data


    return Data, Mask, np.asarray(labelz, dtype='int32'), val_Data, val_Mask, np.asarray(val_labelz, dtype='int32')

Data, Msk, Targets, val_Data, val_Msk, val_tars = load_dataset()


f = h5py.File('dataset.hdf5', mode='w')
data = (('train', 'features', Data),
        ('train', 'mask', Msk),
        ('train', 'targets', Targets),
        ('valid', 'features', val_Data),
        ('valid', 'mask', val_Msk),
        ('valid', 'targets', val_tars))
fill_hdf5_file(f, data)

for i, label in enumerate(('batch', 'maxlen', 'feat_dim')):
    f['features'].dims[i].label = label
for i, label in enumerate(('batch', 'maxlen')):
    f['mask'].dims[i].label = label
for i, label in enumerate(('batch',)):
    f['targets'].dims[i].label = label

f.flush()
f.close()