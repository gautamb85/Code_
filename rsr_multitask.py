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
from utils import load_data

cFile = sys.argv[1]

f=open(cFile)
lines = f.readlines()
lines = [l.strip() for l in lines]
values = [v.split()[1] for v in lines]


N_HIDDEN = int(values[0])
F_DIM = int(values[1])
LEARNING_RATE = np.cast['float32'](values[2])
BATCH_SIZE = int(values[3])
NUM_EPOCHS = int(values[4])
NUM_SPEAKERS = int(values[5])
NUM_PHRASES = int(values[6])
MODEL_NAME = values[7]
SAVE_PATH = values[8]
#data
TRAIN_FILE = values[9]
VALID_FILE = values[10]
SCRIPT = values[11]
MSC = values[12]

wtspth =os.path.join(SAVE_PATH,MODEL_NAME,'weights')
logpth =os.path.join(SAVE_PATH,MODEL_NAME,'logs')

#create a folder for logs and weights
if os.path.exists(wtspth):
    print 'weight folder exits\n'
else:
    Command1 = "mkdir -p" +" "+ wtspth
    process = subprocess.check_call(Command1.split())

if os.path.exists(logpth):
    print 'logs folder exists'
else:
    Command2 = "mkdir -p" +" "+ logpth
    process2 = subprocess.check_call(Command2.split())



# Min/max sequence length
MAX_LENGTH = 800

def build_rnn(net_input=None, mask_input=None):

    print("Building network ...")

    l_in = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, F_DIM), input_var = net_input)
    n_batch,_,_ = l_in.input_var.shape
    #print lasagne.layers.get_output(l_in, inputs={l_in: X}).eval({X: x_dummy}).shape

    l_mask = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH), input_var = mask_input)
    #print lasagne.layers.get_output(l_mask, inputs={l_mask: Mask}).eval({Mask: mask}).shape

    #initialize the gates
    gate_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    b=lasagne.init.Constant(0.))
    
    cell_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    W_cell=None, b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh)
    

    l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, mask_input=l_mask,
                                         precompute_input=True, peepholes=True, 
                                         learn_init=True, only_return_final=True)
        
    
    l_backward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, mask_input=l_mask,
                                         precompute_input=True, peepholes=True, 
                                         learn_init=True, only_return_final=True, backwards=True)
    
    
    
    l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

    l_spk_softmax = lasagne.layers.DenseLayer(l_sum, num_units=NUM_SPEAKERS, nonlinearity=lasagne.nonlinearities.softmax)

    l_phr_softmax = lasagne.layers.DenseLayer(l_sum, num_units=NUM_PHRASES, nonlinearity=lasagne.nonlinearities.softmax)
    
    return l_spk_softmax, l_phr_softmax


#print lasagne.layers.get_output(l_softmax, inputs={l_in: X, l_mask: Mask}).eval({X: d1, Mask: m1}).shape
#l_softmax = lasagne.layers.ReshapeLayer(l_dense, (n_batch, MAX_LENGTH, N_HIDDEN))

def main():
        
    X = T.tensor3(name='input',dtype='float32')
    MASK = T.matrix(name = 'mask', dtype='float32')
    SPK_LABELS = T.ivector(name='spk_labels')
    PHR_LABELS = T.ivector(name='phr_labels')

    #load data
    train_set = H5PYDataset(TRAIN_FILE,  which_sets=('train',))
    valid_set = H5PYDataset(TRAIN_FILE, which_sets=('test',))

    spk_network, phr_network = build_rnn(net_input=X, mask_input= MASK)
    
    network_output_spk = lasagne.layers.get_output(spk_network)
    network_output_phr = lasagne.layers.get_output(phr_network)

    val_prediction_spk = lasagne.layers.get_output(spk_network, deterministic=True)
    val_prediction_phr = lasagne.layers.get_output(phr_network, deterministic=True)
    #needed for accuracy
    
    val_acc_spk = T.mean(T.eq(T.argmax(val_prediction_spk, axis=1), SPK_LABELS), dtype=theano.config.floatX)
    val_acc_phr = T.mean(T.eq(T.argmax(val_prediction_phr, axis=1), PHR_LABELS), dtype=theano.config.floatX)

    #training accuracy
    train_acc_spk = T.mean(T.eq(T.argmax(network_output_spk, axis=1), SPK_LABELS), dtype=theano.config.floatX)
    train_acc_phr = T.mean(T.eq(T.argmax(network_output_phr, axis=1), PHR_LABELS), dtype=theano.config.floatX)
    
    #cost function    
    spk_cost = lasagne.objectives.categorical_crossentropy(network_output_spk, SPK_LABELS)
    phr_cost = lasagne.objectives.categorical_crossentropy(network_output_phr, PHR_LABELS)
    total_cost = spk_cost + phr_cost
    mean_cost = total_cost.mean()
    
    #Validation cost
    val_spk_cost = lasagne.objectives.categorical_crossentropy(val_prediction_spk, SPK_LABELS)
    val_phr_cost = lasagne.objectives.categorical_crossentropy(val_prediction_phr, PHR_LABELS)
    val_cost = val_spk_cost + val_phr_cost
    val_mcost = val_cost.mean()

    #Get parameters of both encoder and decoder
    all_parameters = lasagne.layers.get_all_params([spk_network, phr_network], trainable=True)

    print("Trainable Model Parameters")
    print("-"*40)
    for param in all_parameters:
        print(param, param.get_value().shape)
    print("-"*40)
    #add grad clipping to avoid exploding gradients
    all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
    #all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

    updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=LEARNING_RATE)

    train_func = theano.function([X, MASK, SPK_LABELS, PHR_LABELS], [mean_cost, train_acc_spk, train_acc_phr], updates=updates)

    val_func = theano.function([X, MASK, SPK_LABELS, PHR_LABELS], [val_mcost, val_acc_spk, val_acc_phr])

    trainerr=[]
    epoch=0 #set the epoch counter
    
    min_val_loss = np.inf
    max_val = np.inf*(-1)
    
    patience=0
    
    print("Starting training...")
        # We iterate over epochs:
    while 'true':
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        tr_acc_spk = 0
        tr_acc_phr = 0

        train_batches = 0

        h1=train_set.open()
        h2=valid_set.open()

        scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=BATCH_SIZE)
        scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=32)


        train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
        valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

        start_time = time.time()

        for data in train_stream.get_epoch_iterator():
            t_data, t_mask, t_plabs,_, t_labs,_ = data
            terr, tacc_spk, tacc_phr = train_func(t_data, t_mask, t_labs, t_plabs)
            train_err += terr
            
            tr_acc_spk += tacc_spk
            tr_acc_phr += tacc_phr
            
            train_batches += 1

        val_err = 0
        val_acc_spk = 0
        val_acc_phr = 0
        
        val_batches = 0

        for data in valid_stream.get_epoch_iterator():
            v_data, v_mask, v_ptars, _, v_tars, _ = data
            err, acc_spk, acc_phr = val_func(v_data, v_mask ,v_tars, v_ptars)
            val_err += err
            val_acc_spk += acc_spk
            val_acc_phr += acc_phr
            
            val_batches += 1

        trainerr.append(train_err/train_batches)

        epoch+=1
        train_set.close(h1)
        valid_set.close(h2)
        
        #Display
        if display:
            
            print("Epoch {} of {} took {:.3f}s".format(
            epoch, NUM_EPOCHS, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training accuracy (speaker):\t\t{:.2f} %".format(
                tr_acc_spk / train_batches * 100))
            print("  training accuracy (phrase):\t\t{:.2f} %".format(
                tr_acc_phr / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy (speaker):\t\t{:.2f} %".format(
                val_acc_spk / val_batches * 100))
            print("  validation accuracy (phrase):\t\t{:.2f} %".format(
                val_acc_phr / val_batches * 100))

        logfile = os.path.join(SAVE_PATH,MODEL_NAME,'logs',MODEL_NAME+'.log')
        flog1 = open(logfile,'ab')
        flog1.write("Running %s on %s" % (SCRIPT,MSC))
        flog1.write("Epoch {} of {} took {:.3f}s \n".format(
        epoch, NUM_EPOCHS, time.time() - start_time))
        flog1.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
        flog1.write("  training accuracy (speaker):\t\t{:.2f} %\n".format(
            tr_acc_spk / train_batches * 100))
        flog1.write("  training accuracy (phrase):\t\t{:.2f} %\n".format(
            tr_acc_phr / train_batches * 100))
        flog1.write("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        flog1.write("  validation accuracy (speaker):\t\t{:.2f} %\n".format(
            val_acc_spk / val_batches * 100))
        flog1.write("  validation accuracy (phrase):\t\t{:.2f} %\n".format(
            val_acc_phr / val_batches * 100))
        
        flog1.write("\n")
        flog1.close()

        if epoch == NUM_EPOCHS:
            break
 
        valE = val_err/val_batches
        
        if valE <= min_val_loss:

            model_params1 = lasagne.layers.get_all_param_values([spk_network, phr_network])

            model1_name = MODEL_NAME+'_minloss' + '.pkl'

            vpth1 = os.path.join(SAVE_PATH, MODEL_NAME, 'weights',model1_name)

            fsave = open(vpth1,'wb')  

            cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)

            fsave.close()
            
            patience=0

            min_val_loss = valE

        #Patience / Early stopping
        else:
            patience+=1
        
        if patience==10:
            break
            
    #Save the final model

    print('Saving Model ...')
    model_params = lasagne.layers.get_all_param_values([spk_network, phr_network])
    model1_name = MODEL_NAME+'_final' + '.pkl'
    vpth = os.path.join(SAVE_PATH, MODEL_NAME,'weights',model1_name)
    fsave = open(vpth,'wb')  
    cPickle.dump(model_params, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
    fsave.close()
    
if __name__ == '__main__':
    main()