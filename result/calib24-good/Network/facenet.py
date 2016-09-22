# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 23:09:01 2015

@author: yy374
"""

### Network/facenet64.py

from chainer import FunctionSet, Variable, cuda
from chainer import functions as F
import numpy as np


### NEED TO CHANGE calc-confidence arguments too I think. Think about how to predict. 

class FaceNet12(FunctionSet):
    def __init__(self):
        super(FaceNet12, self).__init__(      
            conv=F.Convolution2D(3, 16, ksize=2, stride=1),
            fc1=F.Linear(400, 16), ## 5x5x16
            fc2=F.Linear(16, 2),
        )
    
    def forward(self, x_data, y_data, train, sub24=False):
        x = Variable(x_data, volatile=not train)
        if train == True and sub24 == False:
            t = Variable(y_data)
        
        h = self.conv(x)
        h = F.relu(F.max_pooling_2d(h, ksize=3, stride=2))
        h = F.relu(self.fc1(h))
        if sub24 == True:
            return h
        else:
            y = F.relu(self.fc2(h))
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x_data): 
        y = self.forward(x_data, None, train=False)
        return F.softmax(y)


class FaceNet24(FunctionSet):
    def __init__(self):
        super(FaceNet24, self).__init__( 
            net12 = FaceNet12(),
            conv=F.Convolution2D(3, 64, ksize=4, stride=1),
            fc1=F.Linear(6400 , 128),  ## 10x10x64
            fc2=F.Linear(144, 2), ## 144 = 128 + 16
        )
    
    def forward(self, x24_data, x12_data, y_data, train, sub48=False):
        x24 = Variable(x24_data, volatile=not train)
        #if  sub48 == False:
         #   x12 = Variable(x12_data, volatile=not train)        
        if train==True and sub48 == False:        
            t = Variable(y_data)
        
        ### 24-net
        h = self.conv(x24)
        h = F.relu(F.max_pooling_2d(h, ksize=3, stride=2))
        h = F.relu(self.fc1(h))
        if sub48 == True:
            return h
        
        ### 12-net substracture layer
        h2 = self.net12.forward(x12_data, None, train, sub24=True)
        #print cuda.to_cpu(h2.data).shape
        # h3 = np.concatenate([cuda.to_gpu(h.data), cuda.to_gpu(h2.data)], axis=1) ## h.data = batchsize x inputsize
        h3 = F.concat((h, h2), axis=1)
        # h3 = h3.astype(np.float32)
        # h3 = Variable(h3, volatile=True) 
        y = F.relu(self.fc2(h3))
        
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x24_data, x12_data):
        y = self.forward(x24_data, x12_data, None, train=False)
        return F.softmax(y)
        

class FaceNet48(FunctionSet):
    def __init__(self):
        super(FaceNet48, self).__init__( 
            net24 = FaceNet24(),
            conv1=F.Convolution2D(3, 64, ksize=4, stride=1),
            ln1=F.LocalResponseNormalization(n=9),
            conv2=F.Convolution2D(64,64, ksize=4, stride=1),
            ln2=F.LocalResponseNormalization(n=9),
            fc1=F.Linear(5184, 256),
            fc2=F.Linear(384, 2), ## 384 = 256 + 128
        )

    
    def forward(self, x48_data, x24_data, y_data, train):
        x48 = Variable(x48_data, volatile=not train)
        # x24 = Variable(x24_data, volatile=not train)
        if train==True:        
            t = Variable(y_data)
        
        ### 48-net
        h = self.conv1(x48)
        h = F.relu(self.ln1(F.max_pooling_2d(h, ksize=3, stride=2)))
        h = self.conv2(h)
        h = F.relu(self.ln2(F.max_pooling_2d(h, ksize=3, stride=2)))
        h = F.relu(self.fc1(h))
        
        ### 24-net substracture layer
        h2 = self.net24.forward(x24_data, None, None, train, sub48=True)
        h3 = F.concat((h, h2), axis=1)
        
        y = F.relu(self.fc2(h3))
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x48_data, x24_data):
        y = self.forward(x48_data, x24_data, None, train=False)
        return F.softmax(y)
        

