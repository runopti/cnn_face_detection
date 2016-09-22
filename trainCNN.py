# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:34:58 2015

@author: yy374
"""

import chainer.functions as F
from chainer import Variable, optimizers, FunctionSet, cuda
from chainer import computational_graph as c

from Network import calibnet

import argparse
import numpy as np
import pickle
import time
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


## Setting
wd_rate = 0.0001
lr_decay = 0.97
n_epoch = 25
model_interval = 5
batchsize = 10
log_interval = 20
valid_interval = 10000

## Load data
data = np.load('data/calib/data_calib_12')
target = np.load('data/calib/target_calib_12')

data_size = len(data)

assert data_size == len(target)

training_size = int(9.0/10*data_size)
perm = np.random.permutation(data_size)
data = data[perm]
target = target[perm]
target = target - 1   ### THIS IS FOR CALIBRATION LABEL
x_train, x_test = np.split(data, [training_size])
y_train, y_test = np.split(target, [training_size]) 

test_size = data_size-training_size

## Load Network Architecture
model = calibnet.CalibNet12()

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

## Set optimizer
optimizer = optimizers.MomentumSGD(lr=0.0001, momentum=0.9)
optimizer.setup(model.collect_parameters())

## function for test phase

def forward_valid(x_test, y_test, batchsize):
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, test_size, batchsize): 
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)
        
        loss, acc = model.forward(x_batch, y_batch, train=True)
        
        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
    
    return sum_loss, sum_accuracy


## Training 
train_plt_loss = []
train_plt_accuracy = []
test_plt_loss = []
test_plt_accuracy = []
epoch_iter = 0
if batchsize > 0:
    epoch_iter = training_size//batchsize+1
begin_at = time.time()

for epoch in range(1, n_epoch+1):
    print "epoch {}".format(epoch)
    train_duration = 0
    
    perm = np.random.permutation(training_size)
    sum_accuracy = 0
    sum_loss = 0
    N = batchsize*log_interval
    sum_plt_accuracy = 0
    sum_plt_loss = 0
    for iter in range(0, training_size, batchsize):
        iter_begin_at = time.time()
        x_batch = x_train[perm[iter:iter + batchsize]] ## take 100 rows of x_train
        y_batch = y_train[perm[iter:iter + batchsize]]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)
        
        # Fills all gradient arrays with zero
        optimizer.zero_grads()
        # Compute error and accuracy rate by feed-forward network
        loss, acc = model.forward(x_batch, y_batch, train=True)
        # Compute 
        loss.backward()
        # Updates all parameters and states using corresponding
        # gradients
        optimizer.weight_decay(wd_rate)
        optimizer.update()
        
        train_duration += time.time() - iter_begin_at
        
        if epoch == 1 and iter == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')
            
        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

        if iter % log_interval == 0:
            if train_duration != 0: ## just scared of zero division..
                throughput = batchsize * iter *1.0 / train_duration
            print('training: iteration={:d}, mean loss={:.8f}, accuracy rate={:.6f}, learning rate={:f}, weight decay={:f}'
                 .format(iter-1+(epoch-1)*epoch_iter, sum_loss/N, sum_accuracy/N,optimizer.lr, wd_rate))
                   
            print('epoch {}: passed time={}, throughput ({} images/sec)'
                .format(epoch, train_duration, throughput))
            sum_plt_loss += sum_loss
            sum_plt_accuracy += sum_accuracy
            sum_loss = 0
            sum_accuracy = 0
            
        if iter % valid_interval == 0:
            valid_begin_at = time.time()
            valid_sum_loss, valid_sum_accuracy = forward_valid(x_test, y_test, batchsize)
            valid_duration = time.time() - valid_begin_at
            if valid_duration != 0:
                throughput = test_size * 1.0 / valid_duration
            print('validation: iteration={:d}, mean loss={:.8f}, accuracy rate={:.6f}'
                .format(iter-1+(epoch-1)*epoch_iter, valid_sum_loss/test_size, valid_sum_accuracy/test_size))
            print('validation time ={}, throughput ({} images/sec)'
                .format(valid_duration, throughput))
            
        sys.stdout.flush()
        
    if sum_loss is not 0:
        sum_plt_loss += sum_loss
        sum_plt_accuracy += sum_accuracy
    train_plt_loss.append(sum_plt_loss / training_size)
    train_plt_accuracy.append(sum_plt_accuracy / training_size)
    
    valid_sum_loss, valid_sum_accuracy = forward_valid(x_test, y_test, batchsize)
    test_plt_loss.append(valid_sum_loss / test_size)
    test_plt_accuracy.append(valid_sum_accuracy / test_size)

    
    optimizer.lr *= lr_decay
    wd_rate *= lr_decay
    
    
    if(epoch % model_interval == 0):
        print('Saving model...(epoch {})'.format(epoch))
        pickle.dump(model, open('model-' + str(epoch) + '.dump', 'wb'), -1)

print('Training finished, total duration={} sec.'.format(time.time()-begin_at))
pickle.dump(model, open('model.dump', 'wb'), -1)

plt.figure()
plt.plot(range(n_epoch), train_plt_loss, label='training')
plt.plot(range(n_epoch), test_plt_loss, label='validation')
plt.legend(loc='upper right')
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("calibnet12_loss.png")

plt.figure()
plt.plot(range(n_epoch), train_plt_accuracy, label='training')
plt.plot(range(n_epoch), test_plt_accuracy, label='validation')
plt.legend(loc='lower right')
plt.title("Classification Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("calibnet12_accuracy.png")






