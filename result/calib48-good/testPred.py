# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:35:04 2015

@author: yy374
"""

from PIL import Image
from chainer import cuda, Variable
import chainer.functions as F
import argparse
import pickle
import numpy as np
from Network import facenet


def prepareImg(im, dim):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    # parser.add_argument('--image', type=str, default='.//face2.img')
    args = parser.parse_args()    
    image_path = 'G:/GraphicsLab/test/img/angle3.jpg'
    im = Image.open(image_path)    
    im24 = im.resize((24,24))
    im12 = im.resize((12,12))
    im24 = prepareImg(im24, 24)
    im12 = prepareImg(im12, 12)
    cnn = pickle.load(open('result/net24/model-25.dump', 'rb'))
    if args.gpu >= 0:
        cuda.init()     
        cnn.to_gpu()
        im24 = cuda.to_gpu(im24)
        im12 = cuda.to_gpu(im12)        


    
    pred = cnn.calc_confidence(im24, im12)
    
    if args.gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data
    
    print("nonface : {} , face : {}".format(pred[0][0], pred[0][1]))