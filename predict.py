# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 00:36:02 2015

@author: yy374
"""

### THIS IS GOING TO BE predict.py, which is called from driver-prediction.py

from Network import facenet
from chainer import cuda
import numpy as np
import pickle
import cv2



def predict_12Net(im, gpu):


    im = cv2.resize(im, (12,12))

    if gpu >= 0:
        cuda.init()
        net12 = pickle.load(open('result/net12/model-15.dump', 'rb')) ## model = facenet.FaceNet12()
        net12.to_gpu()
        
        
        
    image = prepareImage(im, 12)
        

    pred = net12.calc_confidence(image)
    
    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    return pred[0][0], pred[0][1]



def predict_24Net(im, gpu):
    
    im24 = cv2.resize(im, (24,24))
    im12 = cv2.resize(im, (12,12))

    if gpu >= 0:
        cuda.init()
        net24 = pickle.load(open('result/net24-good/model-30.dump', 'rb')) ## model = facenet.FaceNet12()
        net24.to_gpu()

    image24 = prepareImage(im24, 24)
    image12 = prepareImage(im12, 12)
    
    pred = net24.calc_confidence(image24, image12)
    
    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    return pred[0][0], pred[0][1]


def predict_48Net(im, gpu):
    
    im24 = cv2.resize(im, (24,24))
    im48 = cv2.resize(im, (48,48))

    if gpu >= 0:
        cuda.init()
        net48 = pickle.load(open('model_48net.dump', 'rb')) ## model = facenet.FaceNet12()
        net48.to_gpu()

    image24 = prepareImage(im24)
    image48 = prepareImage(im48)
    
    pred = net48.calc_confidence(image48, image24)
    
    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    return pred[0][0], pred[0][1]



def prepareImage(im, dim):

    image = np.asarray(im).astype(np.float32)
 
    if(len(image.shape)==2): ## meaning black and white image
        temp = np.zeros((dim,dim,3))
        # stack the image matrix in the depth dimension
        # creating (64L, 64L, 3L) black and white
        for i in range(3):
            temp[:,:,i] = image
        image = temp
    image = np.transpose(image, (2,0,1))
    # image -= mean # 
    image /= 255 # rescaling to -1 to 1
    image = image.astype(np.float32)
    image = image[np.newaxis, :,:,:]
        
    image = cuda.to_gpu(image.copy())
    return image