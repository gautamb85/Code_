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
import cPickle
import sys
import subprocess
import scipy.io

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from collections import OrderedDict
import h5py

from fuel.datasets import H5PYDataset
from fuel.converters.base import fill_hdf5_file
from utils import load_data

cFile = sys.argv[1]

f=open(cFile)
lines = f.readlines()
lines = [l.strip() for l in lines]
values = [v.split()[1] for v in lines]

N_HIDDEN = int(values[0])
F_DIM = int(values[1])
LEARNING_RATE = np.cast['float32'](values[2])
NUM_EPOCHS = int(values[3])
BATCH_SIZE = int(values[4])
MODEL_NAME = values[5]
SAVE_PATH = values[6]
#data
TRAIN_FILE = values[7]
VALID_FILE = values[8]
SCRIPT = values[9] #script name 
MSC = values[10] #machine

def build_dnn(net_input=None):

    print("Building network ...")

    l_in = lasagne.layers.InputLayer(shape=(None, F_DIM), input_var = net_input)
    n_batch,_ = l_in.input_var.shape
    #print lasagne.layers.get_output(l_in, inputs={l_in: X}).eval({X: x_dummy}).shape

    #print lasagne.layers.get_output(l_mask, inputs={l_mask: Mask}).eval({Mask: mask}).shape
    
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=N_HIDDEN,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    
    l_hid3 = lasagne.layers.DenseLayer(l_hid2_drop, num_units=N_HIDDEN,
                            nonlinearity=lasagne.nonlinearities.rectify)
    
    l_hid4 = lasagne.layers.DenseLayer(l_hid3, 800,
                            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid5 = lasagne.layers.DenseLayer(l_hid4, num_units=400,
                            nonlinearity=lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(
            l_hid5, num_units=F_DIM,
            nonlinearity=lasagne.nonlinearities.linear)
    
    return l_out


def main():
        
    X = T.matrix(name='input_ivecs',dtype='float32')
    
     #load data
    #train_set = H5PYDataset(TRAIN_FILE,  which_sets=('train',))
    valid_set = H5PYDataset(VALID_FILE, which_sets=('valid',))

    network = build_dnn(net_input=X)
    
    dae_ivector = lasagne.layers.get_output(network, deterministic=True)

    feature_extr = theano.function([X], dae_ivector)
    
    h1=train_set.open()
    h2=valid_set.open()

    scheme = SequentialScheme(examples=train_set.num_examples, batch_size=512)
    scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=512)

    train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
    valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

    for data in train_stream.get_epoch_iterator():
        
        t_data, _,t_name = data
        t_ivec = feature_extr(t_data)
        
        for name,ivec in zip(t_name,t_ivec):
            fname = os.path.join(SAVE_PATH,t_name)
            scipy.io.savemat(fname,mdict={'ivec':ivec})


    for data in valid_stream.get_epoch_iterator():
        v_data, _,v_name = data
        v_ivec = feature_extr(v_data)
        
        for name,ivec in zip(v_name,v_ivec):
            fname = os.path.join(SAVE_PATH,v_name)
            scipy.io.savemat(fname,mdict={'ivec':ivec})

if __name__ == '__main__':
    main()