# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:25:36 2015

@author: yy374
"""

import cv2


def pyramid(image, scale):
    minSize = (30,30)
    image_list = []
    while True:
        # Compute the new dimensions of the image and resize it
        height, width, channels = image.shape
        w = int(width * 1.0 / scale)
        ratio = height * 1.0 / width  # height / width ratio
        h = int(w * ratio)
        image = cv2.resize(image, (w , h))
        
        if h < minSize[1] or w < minSize[0]:
            break
        
        image_list.append(image)
    
    return image_list


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    window_list = []
    h, w, c = image.shape
    for y in xrange(0, h, stepSize):
        for x in xrange(0, w, stepSize):
            window_list.append((x, y, image[y:y + windowSize[1], x:x + windowSize[0]]))
    return window_list
    
