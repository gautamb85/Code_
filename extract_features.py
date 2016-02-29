
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

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from collections import OrderedDict
import h5py

from fuel.datasets import H5PYDataset
from fuel.converters.base import fill_hdf5_file

# Min/max sequence length
MAX_LENGTH = 800
F_DIM=40
N_HIDDEN=500
NUM_SPEAKERS=194
NUM_PHRASES=30
#def build_rnn(net_input=None, mask_input=None):

    #print("Building network ...")

X = T.tensor3(name='input',dtype='float32')
MASK = T.matrix(name = 'mask', dtype='float32')
    
l_in = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, F_DIM), input_var = X)
n_batch,_,_ = l_in.input_var.shape
#print lasagne.layers.get_output(l_in, inputs={l_in: X}).eval({X: x_dummy}).shape

l_mask = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH), input_var = MASK)
#print lasagne.layers.get_output(l_mask, inputs={l_mask: Mask}).eval({Mask: mask}).shape

#initialize the gates
gate_parameters = lasagne.layers.recurrent.Gate(
W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
b=lasagne.init.Constant(0.))

cell_parameters = lasagne.layers.recurrent.Gate(
W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
W_cell=None, b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh)


l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, mask_input=l_mask,
                                     precompute_input=True, peepholes=True, learn_init=True)


l_backward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, mask_input=l_mask,
                                     precompute_input=True, peepholes=True, learn_init=True, backwards=True)

l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

l_last = lasagne.layers.SliceLayer(l_sum,-1,1)

#to be ignored
l_spk_softmax = lasagne.layers.DenseLayer(l_last, num_units=NUM_SPEAKERS, nonlinearity=lasagne.nonlinearities.softmax)

l_phr_softmax = lasagne.layers.DenseLayer(l_last, num_units=NUM_PHRASES, nonlinearity=lasagne.nonlinearities.softmax)

#set network weights
Wfile = open('/misc/data15/reco/bhattgau/Rnn/projects/Rvector/rsr_multitask_fbank_500/weights/rsr_multitask_fbank_500_minloss.pkl')
Wts = cPickle.load(Wfile)
lasagne.layers.set_all_param_values([l_spk_softmax, l_phr_softmax], Wts)

Rvectors = lasagne.layers.get_output(l_sum)

l_last = Rvectors[:,-1,:]

feat_extractor = theano.function([X, MASK], [Rvectors, l_last])    

TRAIN_FILE='/misc/data15/reco/bhattgau/Rnn/data/rsr/rsr-p1-eval4.hdf5'
SAVE='/misc/data15/reco/bhattgau/Rnn/data/r-vectors/rsr_multitask_500fb/eval/'

R_VEC_FR=[]
R_VEC_UTT = []

train_set = H5PYDataset(TRAIN_FILE,  which_sets=('eval',))
#valid_set = H5PYDataset(TRAIN_FILE, which_sets=('test',))

h1=train_set.open()

scheme = SequentialScheme(examples=train_set.num_examples, batch_size=1)

train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
#valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

for data in train_stream.get_epoch_iterator():
    t_data, t_mask, t_name = data
    t_name = str(t_name)
    t_name = t_name[2:-2]
    print t_name
    #extract features
    
    frame_f, utt_f = feat_extractor(t_data, t_mask)
    lastP = np.sum(t_mask, axis=1)
    frame_f = frame_f[:,:lastP,:]
    frame_f = np.reshape(frame_f, (frame_f.shape[1], frame_f.shape[2]))
    
    fr_feat = os.path.join(SAVE,'frame',t_name+'.npy')
    utt_feat = os.path.join(SAVE,'Utt',t_name +'.npy')
    
    ffile = open(fr_feat, 'wb')
    ufile = open(utt_feat, 'wb')
    
    np.save(ffile, frame_f)
    np.save(ufile, utt_f)

    ffile.close()
    ufile.close()
