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

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from collections import OrderedDict

spth = '/misc/data15/reco/bhattgau/Rnn/projects/Rvector/Weights/basic-softmax'

def load_dataset():
    
    f1 = open('/misc/data15/reco/bhattgau/Rnn/lists/spk_softmax/Train_feats_labs.plst')
    lines = f1.readlines()
    lines = [l.strip() for l in lines]
    labelz = [int(l.split()[1]) for l in lines] 
    #labelz = labelz[:20]
    features = [l.split()[0] for l in lines] 
    
    f2 = open('/misc/data15/reco/bhattgau/Rnn/lists/spk_softmax/Valid_feats_labs.plst')
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


# Min/max sequence length
MAX_LENGTH = 800
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 500
F_DIM = 20
# Number of training sequences ain each batch
N_BATCH = 1
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10

nSpk = 98

X = T.tensor3(name='input',dtype='float32')
Mask = T.matrix(name = 'mask', dtype='float32')

target = T.matrix(name='target_values', dtype='float32')

print("Building network ...")

l_in = lasagne.layers.InputLayer(shape=(None, None, F_DIM), input_var = X)
n_batch,maxlen,_ = l_in.input_var.shape

#get the batch size for the weights matrix
l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var = Mask)
#print lasagne.layers.get_output(l_mask, inputs={l_mask: Mask}).eval({Mask: mask}).shape
#initialize the gates

#Compute Recurrent Embeddings
l_forward = lasagne.layers.GRULayer(l_in, N_HIDDEN, mask_input=l_mask)
l_backward = lasagne.layers.GRULayer(l_in, N_HIDDEN, mask_input=l_mask, backwards=True)
l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

Recc_emb =  lasagne.layers.get_output(l_sum, inputs={l_in: X, l_mask: Mask})#.eval({X: x_dummy, Mask: mask})
#print lasagne.layers.get_output(l_res, inputs={l_in: X, l_mask: Mask}).eval({X: x_dummy, Mask: mask}).shape

l_reshape1 = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))

l_attend = lasagne.layers.DenseLayer(l_reshape1, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
#print lasagne.layers.get_output(l_attend, inputs={l_in: X, l_mask: Mask}).eval({X: x_dummy, Mask: mask}).shape

lstep = T.sum(Mask, axis=1)#.eval({Mask : mask})
lstep = T.cast(lstep, 'int32')#.eval()
lstep = lstep.T

l_attention = lasagne.layers.ReshapeLayer(l_attend, (n_batch, maxlen))
wts = lasagne.layers.get_output(l_attention, inputs={l_in: X, l_mask: Mask})
Wts = T.zeros([n_batch, maxlen], dtype='float32')

AWts = T.zeros([n_batch, maxlen], dtype='float32')

def weighted_avg(w_t, l_t, W, aW, R_emb):
    W = T.set_subtensor(W[:l_t], w_t[:l_t])
    W = W[None,:]
    attn_probs = T.nnet.softmax(W.nonzero_values())
    attn = T.flatten(attn_probs)
    attn = attn.T

    aW = T.set_subtensor(aW[:l_t], attn)
    #aW = aW[None,:]
    
    w_sum = T.dot(aW, R_emb)
    
    #this W (1,600) is is then multiplied (dot product) with the recurrent embedding (600, 100)
    #to produce a feature vector that is a weighted sum of all the embedding vectors of the recording
    #also return the weights for analysis
    
    return w_sum

U_t, _ = theano.scan(fn=weighted_avg, sequences=[wts, lstep, Wts, AWts, Recc_emb])

l_in2 = lasagne.layers.InputLayer(shape=(None,N_HIDDEN), input_var = U_t)
n_batch1,_ = l_in2.input_var.shape

#l_reshape2 = lasagne.layers.ReshapeLayer(l_in2, (n_batch1*l1,N_HIDDEN))
#print lasagne.layers.get_output(l_reshape2, inputs={l_in: X, l_mask: Mask, l_in2: U_t}).eval({X: x_dummy, Mask: mask}).shape

#Finally this feature vector(s) gets passed to a dense layer which represents a softmax distribution over speakers
l_Spk_softmax = lasagne.layers.DenseLayer(l_in2, num_units=98, nonlinearity=lasagne.nonlinearities.softmax)
#print lasagne.layers.get_output(l_Spk_softmax, inputs={l_in: X, l_mask: Mask, l_in2: U_t}).eval({X: x_dummy, Mask: mask}).shape
# lasagne.layers.get_output produces a variable for the output of the net

network_output = lasagne.layers.get_output(l_Spk_softmax)

labels = T.ivector(name='labels')

val_prediction = lasagne.layers.get_output(l_Spk_softmax, deterministic=True)
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
#total_cost = -(labels*T.log(network_MMMmoutput) + (1-labels)*T.log(1-network_output)) 
mean_cost = total_cost.mean()
#accuracy function
val_cost = lasagne.objectives.categorical_crossentropy(val_prediction, labels)
val_mcost = val_cost.mean()

#Get parameters of both encoder and decoder
params1 = lasagne.layers.get_all_params([l_attend], trainable=True)
params2 = lasagne.layers.get_all_params([l_Spk_softmax], trainable=True)
all_parameters = params1 + params2

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

train_set = IndexableDataset(
    indexables = OrderedDict([('features', Data), ('mask',Msk), ('targets', Targets)]), 
    axis_labels={'features':('batch','maxlen','feat_dim'),'mask':('batch','maxlen'), 'targets':('batch','index')})

valid_set = IndexableDataset(
    indexables = OrderedDict([('features', val_Data), ('mask', val_Msk), ('targets', val_tars)]), 
    axis_labels={'features':('batch','maxlen','feat_dim'),'mask':('batch','maxlen'), 'targets':('batch','index')})

num_epochs=300
epoch=0

print("Starting training...")
    # We iterate over epochs:
val_prev = np.inf
a_prev = -np.inf

#for epoch in range(num_epochs):
while('true'):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    tr_acc = 0
    train_batches = 0
    
    t_state = train_set.open()
    v_state = valid_set.open()

    scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=128)
    scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=128)


    train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
    valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)
    
    start_time = time.time()
    
    for data in train_stream.get_epoch_iterator():
        t_data, t_mask, t_labs = data
        terr, tacc = train_func(t_data, t_mask, t_labs)
        train_err += terr
        tr_acc += tacc
        train_batches += 1
        
    val_err = 0
    val_acc = 0
    val_batches = 0
    
    for data in valid_stream.get_epoch_iterator():
        v_data, v_mask, v_tars = data
        err, acc = val_func(v_data, v_mask ,v_tars)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    #f_log = open('/misc/data15/reco/bhattgau/Rnn/Projects/Rvector/Weights/training.log','a')
    epoch+=1

# Then we print the results for this epoch:
    #flog = open('training.log','a')
    print("Epoch {} of {} took {:.3f}s ".format(
    epoch, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f} ".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} % ".format(
        tr_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f} ".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} % ".format(
        val_acc / val_batches * 100))
    #flog.write('\n')
    #flog.close()
   
    valE = val_err/val_batches
    valA = val_acc / val_batches
    
    #save the max accuracy model
    max_val = a_prev
    
    #save model with highest accuracy
    if valA > max_val:
        
        model_params1 = lasagne.layers.get_all_param_values(l_attention)
        model_params2 = lasagne.layers.get_all_param_values(l_Spk_softmax)

        model1_name = 'Attn_acc' + '.pkl'
        model2_name = 'Attn_accB' + '.pkl'

        vpth1 = os.path.join(spth, model1_name)
        vpth2 = os.path.join(spth, model2_name)

        fsave = open(vpth1,'wb')  
        fsave2 = open(vpth2,'wb')  

        cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(model_params2, fsave2, protocol=cPickle.HIGHEST_PROTOCOL)
        fsave.close()
        fsave2.close()
        
        max_val = valA

    #Patience

    if valE > val_prev:
        c+=1
        
        #save the model incase
        model_params1 = lasagne.layers.get_all_param_values(l_attention)
        model_params2 = lasagne.layers.get_all_param_values(l_Spk_softmax)

        model1_name = 'Attn_ofit'  + '.pkl'
        model2_name = 'Attn_ofitB' + '.pkl'

        vpth1 = os.path.join(spth, model1_name)
        vpth2 = os.path.join(spth, model2_name)

        fsave = open(vpth1,'wb')  
        fsave2 = open(vpth2,'wb')  

        cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(model_params2, fsave2, protocol=cPickle.HIGHEST_PROTOCOL)
        fsave.close()
        fsave2.close()
        
        val_prev=valE
        
    else:
        c=0
        val_prev=valE
    
    if c==8:
        break
    
    if epoch==num_epochs:
        break
        
#Save the final model


print('Saving Model ...')
model_params1 = lasagne.layers.get_all_param_values(l_attention)
model_params2 = lasagne.layers.get_all_param_values(l_Spk_softmax)

model1_name = 'Attn_softmax_final' + '.pkl'
model2_name = 'Attn_softmax_final' + '.pkl'

vpth1 = os.path.join(spth, model1_name)
vpth2 = os.path.join(spth, model2_name)

fsave = open(vpth1,'wb')  
fsave2 = open(vpth2,'wb')  

cPickle.dump(model_params1, fsave, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(model_params2, fsave2, protocol=cPickle.HIGHEST_PROTOCOL)


fsave.close()
fsave2.close()
