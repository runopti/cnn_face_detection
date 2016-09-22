# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:29:50 2015

@author: yy374
"""
 
### calibnet

from chainer import FunctionSet, Variable
from chainer import functions as F



class CalibNet12(FunctionSet):
    def __init__(self):
        super(CalibNet12, self).__init__(  
            conv=F.Convolution2D(3, 16, ksize=2, stride=1),
            fc1=F.Linear(400, 128),
            fc2=F.Linear(128, 45),
        )
    
    def forward(self, x_data, y_data, train):
        x = Variable(x_data, volatile=not train)
        if train == True:        
            t = Variable(y_data)
        
        h = self.conv(x)
        h = F.relu(F.max_pooling_2d(h, ksize=3, stride=2))
        h = F.relu(self.fc1(h))
        y = F.relu(self.fc2(h))
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x_data): 
        y = self.forward(self, x_data, None, train=False)
        return F.softmax(y)


class CalibNet24(FunctionSet):
    def __init__(self):
        super(CalibNet24, self).__init__(  
            conv=F.Convolution2D(3, 64, ksize=4, stride=1),
            fc1=F.Linear(6400, 64),
            fc2=F.Linear(64, 45), 
        )
    
    def forward(self, x24_data, y_data, train, sub48=False):
        x24 = Variable(x24_data, volatile=not train)      
        if train == True:        
            t = Variable(y_data)
        
        ### 24-net
        h = self.conv(x24)
        h = F.relu(F.max_pooling_2d(h, ksize=3, stride=2))
        h = F.relu(self.fc1(h))
        y = F.relu(self.fc2(h))
        
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x24_data):
        y = self.forward(self, x24_data, None, train=False)
        return F.softmax(y)
        

class CalibNet48(FunctionSet):
    def __init__(self):
        super(CalibNet12, self).__init__(  
            conv1=F.Convolution2D(3, 64, ksize=4, stride=1),
            ln1=F.LocalResponseNormalization(n=9),
            conv2=F.Convolution2D(64,64, ksize=4, stride=1),
            ln2=F.LocalResponseNormalization(n=9),
            fc1=F.Linear(23104, 256),
            fc2=F.Linear(256, 45), ## 384 = 256 + 128
        )
    
    def forward(self, x48_data, y_data, train):
        x48 = Variable(x48_data, volatile=not train)
        if train == True:        
            t = Variable(y_data)
        
        ### 48-net
        h = self.conv1(x48)
        h = self.ln1(F.relu(F.max_pooling_2d(h, ksize=3, stride=2)))
        h = self.ln2(self.conv2(h))
        h = F.relu(self.fc1(h))
        y = F.relu(self.fc2(h))
        
        if train == True:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            return y 
    
    def calc_confidence(self, x48_data):
        y = self.forward(self, x48_data, None, train=False)
        return F.softmax(y)
        

