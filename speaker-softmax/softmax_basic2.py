import htkmfc
import os
import lasagne
import time
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle

from IPython import display

def load_dataset():
    
    f1 = open('/misc/data15/reco/bhattgau/Rnn/Lists/spk_softmax/Train_feats_labs.plst')
    lines = f1.readlines()
    lines = [l.strip() for l in lines]
    labelz = [int(l.split()[1]) for l in lines] 
    #labelz = labelz[:20]
    features = [l.split()[0] for l in lines] 
    
    f2 = open('/misc/data15/reco/bhattgau/Rnn/Lists/spk_softmax/Valid_feats_labs.plst')
    lines = f2.readlines()
    lines = [l.strip() for l in lines]
    val_labelz = [int(l.split()[1]) for l in lines] 
    #val_labelz = val_labelz[:20]
    val_features = [l.split()[0] for l in lines] 
    
    n_samp = len(features)
    maxlen=800 #pad all utterances to this length
    feat_dim=20
    nSpk = 98
    dpth = '/misc/data15/reco/bhattgau/Rnn/Data/mfcc/Nobackup/VQ_VAD_HO_EPD/'

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

def iterate_minibatches(inputs, mask, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    iplen = len(inputs)
    pointer = 0
    indices=np.arange(iplen)
    
    if shuffle:
        np.random.shuffle(indices)
    
    while pointer < iplen:
    
        if pointer <= iplen - batchsize:
            excerpt = indices[pointer:pointer+batchsize]
        else:
            batchsize = iplen-pointer
            excerpt = indices[pointer:pointer+batchsize]
        
        pointer+=batchsize

        yield inputs[excerpt], mask[excerpt], targets[excerpt]



# Min/max sequence length
MAX_LENGTH = 800
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 400
F_DIM = 20
# Number of training sequences ain each batch
N_BATCH = 1
# Optimization learning rate
# All gradients above this will be clipped
nSpk = 98

X = T.tensor3(name='input',dtype='float32')
Mask = T.matrix(name = 'mask', dtype='float32')

target = T.matrix(name='target_values', dtype='float32')

print("Building network ...")

l_in = lasagne.layers.InputLayer(shape=(None, None, F_DIM), input_var = X)
n_batch,_,_ = l_in.input_var.shape
#print lasagne.layers.get_output(l_in, inputs={l_in: X}).eval({X: x_dummy}).shape

l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var = Mask)
#print lasagne.layers.get_output(l_mask, inputs={l_mask: Mask}).eval({Mask: mask}).shape

#initialize the gates
l_forward = lasagne.layers.GRULayer(l_in, N_HIDDEN, precompute_input=True,  mask_input=l_mask)
l_backward = lasagne.layers.GRULayer(l_in, N_HIDDEN, precompute_input=True, mask_input=l_mask,
                                    backwards=True)
l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

l_forward1 = lasagne.layers.GRULayer(l_sum, 200, precompute_input=True, only_return_final=True,mask_input=l_mask)
l_backward1 = lasagne.layers.GRULayer(l_sum, 200, precompute_input=True, mask_input=l_mask, only_return_final=True,
                                    backwards=True)

l_sum1 = lasagne.layers.ConcatLayer([l_forward1, l_backward1])

#l_forward2 = lasagne.layers.GRULayer(l_sum1, 100, precompute_input=True, mask_input=l_mask, only_return_final=True)
#l_backward2 = lasagne.layers.GRULayer(l_sum1, 100, precompute_input=True, mask_input=l_mask, only_return_final=True,
                                    #backwards=True)
#l_concat = lasagne.layers.ConcatLayer([l_forward2, l_backward2])

#l_proj = lasagne.layers.DenseLayer(l_sum, num_units=100, nonlinearity=lasagne.nonlinearities.linear)

l_softmax = lasagne.layers.DenseLayer(l_sum1, num_units=nSpk, nonlinearity=lasagne.nonlinearities.softmax)
#print lasagne.layers.get_output(l_softmax, inputs={l_in: X, l_mask: Mask}).eval({X: d1, Mask: m1}).shape
labels = T.ivector(name='labels')

network_output = lasagne.layers.get_output(l_softmax)
val_prediction = lasagne.layers.get_output(l_softmax, deterministic=True)
#needed for accuracy
#don't use the one hot vectors here
#The one hot vectors are needed for the categorical cross-entropy
val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), labels), dtype=theano.config.floatX)
#training accuracy
train_acc = T.mean(T.eq(T.argmax(network_output, axis=1), labels), dtype=theano.config.floatX)


#T.argmax(network_output, axis=1).eval({X: d1, Mask: m1})

#print network_output.eval({X: d1, Mask: m1})[1][97]
#cost function
total_cost = lasagne.objectives.categorical_crossentropy(network_output, labels)
#total_cost = -(labels*T.log(network_output) + (1-labels)*T.log(1-network_output)) 
mean_cost = total_cost.mean()
#accuracy function
val_cost = lasagne.objectives.categorical_crossentropy(val_prediction, labels)
val_mcost = val_cost.mean()

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params([l_softmax], trainable=True)

print("Trainable Model Parameters")
print("-"*40)
for param in all_parameters:
    print(param, param.get_value().shape)
print("-"*40)
#add grad clipping to avoid exploding gradients
all_grads = [T.clip(g,-5,5) for g in T.grad(mean_cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,5)

updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.001)

train_func = theano.function([X, Mask, labels], [mean_cost, train_acc], updates=updates)

val_func = theano.function([X, Mask, labels], [val_mcost, val_acc])


#load the dataset
Data, Msk, Targets, val_Data, val_Msk, val_tars = load_dataset()


num_epochs=500
epoch=0

print("Starting training...")
    # We iterate over epochs:
val_prev = np.inf

#for epoch in range(num_epochs):
while('true'):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    tr_acc = 0
    train_batches = 0
    
    start_time = time.time()
    for batch in iterate_minibatches(Data, Msk, Targets, 128):
        t_data, t_mask, t_labs = batch
        terr, tacc = train_func(t_data, t_mask, t_labs)
        train_err += terr
        tr_acc += tacc
        train_batches += 1
        
    val_err = 0
    val_acc = 0
    val_batches = 0
    
    for batch in iterate_minibatches(val_Data, val_Msk, val_tars, 64, shuffle=False):
        v_data, v_mask, v_tars = batch
        err, acc = val_func(v_data, v_mask ,v_tars)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    #f_log = open('/misc/data15/reco/bhattgau/Rnn/Projects/Rvector/Weights/training.log','a')
    epoch+=1

# Then we print the results for this epoch:
    flog = open('/misc/data15/reco/bhattgau/Rnn/Code/notebooks/speaker-softmax/training.log','a')
    flog.write("Epoch {} of {} took {:.3f}s \n".format(
    epoch, num_epochs, time.time() - start_time))
    flog.write("  training loss:\t\t{:.6f} \n".format(train_err / train_batches))
    flog.write("  training accuracy:\t\t{:.2f} % \n".format(
        tr_acc / train_batches * 100))
    flog.write("  validation loss:\t\t{:.6f} \n".format(val_err / val_batches))
    flog.write("  validation accuracy:\t\t{:.2f} % \n".format(
        val_acc / val_batches * 100))
    flog.write('\n')
    flog.close()
   
    valE = val_err/val_batches
    if valE > val_prev:
        c+=1
        val_prev=valE
    else:
        c=0
        val_prev=valE
    
    if c==5:
        break
    
    if epoch==num_epochs:
        break
        
#Save the final model
spth = '/misc/data15/reco/bhattgau/Rnn/Projects/Rvector/Weights/basic-softmax'

print('Saving Model ...')
model_params = lasagne.layers.get_all_param_values(l_softmax)
model_name = 'basic_softmax_507' + '.pkl'
vpth = os.path.join(spth, model_name)
fsave = open(vpth,'wb')  
cPickle.dump(model_params, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
fsave.close()
