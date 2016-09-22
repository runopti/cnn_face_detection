# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:38:09 2015

@author: yy374
"""

### calibration.pred_calibNet12

from chainer import cuda
import numpy as np
import pickle
import cv2



def pred_calibNet12(box, gpu):
    x, y, winW, winH, window = box
    
    if gpu >= 0:
        cuda.init()
        net12 = pickle.load(open('result/calib12-good/model-15.dump', 'rb')) ## model = facenet.FaceNet12()
        net12.to_gpu()
    
    im = cv2.resize(window, (12,12))
    image = prepareImage(im, 12)
    
    pred = net12.calc_confidence(image)

    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    xn, yn, sn = set_calib_number(pred)
    
    x = x - xn*winW  # I didn't divide this with sn, since this is already done in the training phase
    y = y - yn*winH # the same reason 
    winW = winW / sn
    winH = winH / sn
    
    return x, y, winW, winH
    ## Next thing to do : check this in testPredCalib.py
    # return pred

def pred_calibNet24(box, gpu):
    x, y, winW, winH, window = box
    
    if gpu >= 0:
        cuda.init()
        net24 = pickle.load(open('result/calib24-good/model-25.dump', 'rb')) ## model = facenet.FaceNet12()
        net24.to_gpu()
    
    im = cv2.resize(window, (24,24))
    image = prepareImage(im, 24)
    
    pred = net24.calc_confidence(image)

    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    xn, yn, sn = set_calib_number(pred)
    
    x = x - xn*winW  # I didn't divide this with sn, since this is already done in the training phase
    y = y - yn*winH # the same reason 
    winW = winW / sn
    winH = winH / sn
    
    return x, y, winW, winH
    ## Next thing to do : check this in testPredCalib.py
    # return pred


def set_calib_number(pred):
    calib_dic = np.load("G://GraphicsLab//paper_model//calib_test//calib_info")
    top3 = np.arange(3)
    top3[0] = pred[0].argmax() # getting the most confident class
    pred[0][top3[0]] = 0.0
    top3[1] = pred[0].argmax() # getting the second most
    pred[0][top3[1]] = 0.0
    top3[2] = pred[0].argmax() # getting the thrid most

    temp = 0
    for i in range(3):  # averaging
        temp += calib_dic[top3[i]]
    temp /= 3
    #temp = calib_dic[top3[0]]
    return temp[1], temp[2], temp[3]






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