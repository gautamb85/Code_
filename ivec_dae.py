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
NUM_EPOCHS = int(values[3])
BATCH_SIZE = int(values[4])
MODEL_NAME = values[5]
SAVE_PATH = values[6]
#data
TRAIN_FILE = values[7]
VALID_FILE = values[8]
SCRIPT = values[9] #script name 
MSC = values[10] #machine

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
    X_mean = T.matrix(name = 'target_ivecs', dtype='float32')
    
    #load data
    train_set = H5PYDataset(TRAIN_FILE,  which_sets=('train',))
    valid_set = H5PYDataset(VALID_FILE, which_sets=('valid',))

    network = build_dnn(net_input=X)
    
    network_output = lasagne.layers.get_output(network)

    val_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    #needed for accuracy
    #val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), LABELS), dtype=theano.config.floatX)
    #training accuracy
    #train_acc = T.mean(T.eq(T.argmax(network_output, axis=1), LABELS), dtype=theano.config.floatX)

    #T.argmax(network_output, axis=1).eval({X: d1, Mask: m1})

    #print network_output.eval({X: d1, Mask: m1})[1][97]
    #cost function
    total_cost = lasagne.objectives.squared_error(network_output, X_mean)
    mean_cost = total_cost.mean()
    #accuracy function
    val_cost = lasagne.objectives.squared_error(val_prediction, X_mean)
    val_mcost = val_cost.mean()

    #Get parameters of both encoder and decoder
    all_parameters = lasagne.layers.get_all_params([network], trainable=True)

    print("Trainable Model Parameters")
    print("-"*40)
    for param in all_parameters:
        print(param, param.get_value().shape)
    print("-"*40)
    #add grad clipping to avoid exploding gradients
    
    updates = lasagne.updates.adadelta(mean_cost, all_parameters)

    train_func = theano.function([X, X_mean], mean_cost, updates=updates)

    val_func = theano.function([X, X_mean], val_mcost)

    trainerr=[]
    epoch=0 #set the epoch counter
    
    min_val_loss = np.inf
    patience=0
    
    print("Starting training...")
        # We iterate over epochs:
    while 'true':
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0

        h1=train_set.open()
        h2=valid_set.open()

        scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=BATCH_SIZE)
        scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=64)


        train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
        valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

        start_time = time.time()

        for data in train_stream.get_epoch_iterator():
            
            t_data, t_targ,_ = data
            terr = train_func(t_data, t_targ)
            train_err += terr
            train_batches += 1

        val_err = 0
        val_batches = 0

        for data in valid_stream.get_epoch_iterator():
          
          v_data, v_tars,_ = data    
          err = val_func(v_data, v_tars)
          val_err += err
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
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            

        logfile = os.path.join(SAVE_PATH,MODEL_NAME,'logs',MODEL_NAME+'.log')
        flog1 = open(logfile,'ab')
        
        flog1.write('Running %s on %s' % (SCRIPT, MSC))
        
        flog1.write("Epoch {} of {} took {:.3f}s\n ".format(
        epoch, NUM_EPOCHS, time.time() - start_time))
        flog1.write("  training loss:\t\t{:.6f} ".format(train_err / train_batches))
        flog1.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))

        flog1.write("\n")
        flog1.close()

        if epoch == NUM_EPOCHS:
            break

        valE = val_err/val_batches

        #Patience
        if valE <= min_val_loss:

            model_params1 = lasagne.layers.get_all_param_values(network)

            model1_name = MODEL_NAME+'_minloss' + '.pkl'

            vpth1 = os.path.join(SAVE_PATH, MODEL_NAME, 'weights',model1_name)

            fsave = open(vpth1,'wb')  

            cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)

            fsave.close()

            min_val_loss = valE
            
            patience=0

        #Patience / Early stopping
        else:
            patience+=1
        
        if patience==10:
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
