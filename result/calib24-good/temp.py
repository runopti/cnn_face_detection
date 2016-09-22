# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:28:41 2015

@author: yy374
"""
from sklearn.datasets import fetch_mldata
import chainer.functions as F
from chainer import Variable, optimizers, FunctionSet, cuda
from chainer import computational_graph as c

from Network import facenet

import argparse
import numpy as np
import pickle
import time
import sys
import matplotlib.pyplot as plt

train_plt_loss = pickle.load(open('train_plt_loss', 'rb'))
test_plt_loss = pickle.load(open('test_plt_loss','rb'))
train_plt_accuracy = pickle.load(open('train_plot_accuracy', 'rb'))
test_plt_accuracy = pickle.load(open('test_plt_accuracy', 'rb'))

print len(train_plt_loss)
print len(test_plt_loss)
print len(train_plt_accuracy)
print len(test_plt_accuracy)

plt.figure()
plt.plot(range(1, 26), train_plt_loss, label='training')
plt.plot(range(1, 26), test_plt_loss, label='validation')
plt.legend(loc='upper right')
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("net_loss.png")

plt.figure()
plt.plot(range(1, 26), train_plt_accuracy, label='training')
plt.plot(range(1, 26), test_plt_accuracy, label='validation')
plt.legend(loc='lower right')
plt.title("Classification Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("net_accuracy.png")

### Load data
#data = np.load('data/calib/data_calib_12')
#target = np.load('data/calib/target_calib_12')
#target2 = target - 1

#data_size = len(data)

#assert data_size == len(target)


#print 'fetch MNIST dataset'
#mnist = fetch_mldata('MNIST original')
## mnist.data : 70,000件の784次元ベクトルデータ
#mnist.data   = mnist.data.astype(np.float32)
#mnist.data  /= 255     # 0-1のデータに変換
#
## mnist.target : 正解データ（教師データ）
#mnist.target = mnist.target.astype(np.int32)
