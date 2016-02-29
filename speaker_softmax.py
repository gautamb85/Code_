
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
NUM_SPEAKERS = int(values[4])
MODEL_NAME = values[5]
SAVE_PATH = values[6]
#data
TRAIN_FILE = values[7]
VALID_FILE = values[8]


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
    l_forward = lasagne.layers.GRULayer(l_in, N_HIDDEN, precompute_input=True, mask_input=l_mask, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in, N_HIDDEN, precompute_input=True, mask_input=l_mask, only_return_final=True,
                                        backwards=True)
    l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward])

    #l_proj = lasagne.layers.DenseLayer(l_concat, num_units=64, nonlinearity=lasagne.nonlinearities.linear)

    l_softmax = lasagne.layers.DenseLayer(l_sum, num_units=NUM_SPEAKERS, nonlinearity=lasagne.nonlinearities.softmax)
    #print lasagne.layers.get_output(l_softmax, inputs={l_in: X, l_mask: Mask}).eval({X: d1, Mask: m1}).shape
    return l_softmax

    
    

def main():
        
    X = T.tensor3(name='input',dtype='float32')
    MASK = T.matrix(name = 'mask', dtype='float32')
    LABELS = T.ivector(name='labels')
    
    #load data
    train_set = H5PYDataset(TRAIN_FILE,  which_sets=('train',))
    valid_set = H5PYDataset(VALID_FILE, which_sets=('test',))
    
    network = build_rnn(net_input=X, mask_input= MASK)
    
    network_output = lasagne.layers.get_output(network)

    val_prediction = lasagne.layers.get_output(network, deterministic=True)
    #needed for accuracy
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), LABELS), dtype=theano.config.floatX)
    #training accuracy
    train_acc = T.mean(T.eq(T.argmax(network_output, axis=1), LABELS), dtype=theano.config.floatX)

    #T.argmax(network_output, axis=1).eval({X: d1, Mask: m1})

    #print network_output.eval({X: d1, Mask: m1})[1][97]
    #cost function
    total_cost = lasagne.objectives.categorical_crossentropy(network_output, LABELS)
    #total_cost = -(labels*T.log(network_output) + (1-labels)*T.log(1-network_output)) 
    mean_cost = total_cost.mean()
    #accuracy function
    val_cost = lasagne.objectives.categorical_crossentropy(val_prediction, LABELS)
    val_mcost = val_cost.mean()

    #Get parameters of both encoder and decoder
    all_parameters = lasagne.layers.get_all_params([network], trainable=True)

    print("Trainable Model Parameters")
    print("-"*40)
    for param in all_parameters:
        print(param, param.get_value().shape)
    print("-"*40)
    #add grad clipping to avoid exploding gradients
    all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

    updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=LEARNING_RATE)

    train_func = theano.function([X, MASK, LABELS], [mean_cost, train_acc], updates=updates)

    val_func = theano.function([X, MASK, LABELS], [val_mcost, val_acc])

    trainerr=[]
    epoch=0 #set the epoch counter
    
    val_prev = np.inf
    a_prev = -np.inf

    print("Starting training...")
        # We iterate over epochs:
    while 'true':
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        tr_acc = 0
        train_batches = 0

        h1=train_set.open()
        h2=valid_set.open()

        scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=128)
        scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=32)


        train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
        valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

        start_time = time.time()

        for data in train_stream.get_epoch_iterator():
            t_data, t_mask, _, t_labs = data
            terr, tacc = train_func(t_data, t_mask, t_labs)
            train_err += terr
            tr_acc += tacc
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0

        for data in valid_stream.get_epoch_iterator():
            v_data, v_mask, _, v_tars = data
            err, acc = val_func(v_data, v_mask ,v_tars)
            val_err += err
            val_acc += acc
            val_batches += 1

        trainerr.append(train_err/train_batches)

        epoch+=1
        train_set.close(h1)
        valid_set.close(h2)
        
        #Display
        if display:
            
            print("Epoch {} of {} took {:.3f}s ".format(
            epoch, NUM_EPOCHS, time.time() - start_time))
            print("  training loss:\t\t{:.6f} ".format(train_err / train_batches))
            print("  training accuracy:\t\t{:.2f} % ".format(
                    tr_acc / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} % ".format(
                    val_acc / val_batches * 100))

        logfile = os.path.join(SAVE_PATH,MODEL_NAME,'logs',MODEL_NAME+'.log')
        flog1 = open(logfile,'ab')

        flog1.write("Epoch {} of {} took {:.3f}s\n ".format(
        epoch, NUM_EPOCHS, time.time() - start_time))
        flog1.write("  training loss:\t\t{:.6f} ".format(train_err / train_batches))
        flog1.write("  training accuracy:\t\t{:.2f} %\n ".format(
            tr_acc / train_batches * 100))
        flog1.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
        flog1.write("  validation accuracy:\t\t{:.2f} %\n ".format(
            val_acc / val_batches * 100))
        flog1.write("\n")
        flog1.close()

        if epoch == NUM_EPOCHS:
            break

        valE = val_err/val_batches
        valA = val_acc / val_batches

        #save the max accuracy model
        max_val = a_prev

        #save model with highest accuracy
        if valA > max_val:

            model_params1 = lasagne.layers.get_all_param_values(network)

            model1_name = MODEL_NAME+'_acc' + '.pkl'

            vpth1 = os.path.join(SAVE_PATH, MODEL_NAME, 'weights',model1_name)

            fsave = open(vpth1,'wb')  

            cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)

            fsave.close()

            max_val = valA

        #Patience

        if valE > val_prev:
            c+=1

            #save the model incase
            model_params1 = lasagne.layers.get_all_param_values(network)

            model1_name = MODEL_NAME + '_ofit'  + '.pkl'

            vpth1 = os.path.join(SAVE_PATH, MODEL_NAME, 'weights',model1_name)

            fsave = open(vpth1,'wb')  

            cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)

            fsave.close()

            val_prev=valE

        else:
            c=0
            val_prev=valE

        if c==5:
            break

    #Save the final model

    print('Saving Model ...')
    model_params = lasagne.layers.get_all_param_values(network)
    model1_name = MODEL_NAME+'_final' + '.pkl'
    vpth = os.path.join(SAVE_PATH, MODEL_NAME,'weights',model1_name)
    fsave = open(vpth,'wb')  
    cPickle.dump(model_params, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
    fsave.close()
    
if __name__ == '__main__':
    main()
