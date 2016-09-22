# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:55:16 2015

@author: yy374
"""

import cv2
import helper
import time
import predict
import calibration
# import nms
image_path = "GraphicsLab/test//img//test1.jpg"
#image = Image.open(image_path)
image = cv2.imread(image_path)

w, h, d = image.shape
winW = int(min(w,h)*1.0/10)
winH = winW
stepSize = int(winW*1.0 / 4)

face_candidate = []



for resized in helper.pyramid(image, scale=1.5):
    for (x, y, window) in helper.sliding_window(resized, stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        nonface, face = predict.predict_12Net(window, 0)
    
        if(face > 0.999):
            x, y, winW, winH = calibration.pred_calibNet12((x, y, winW, winH, window), 0)
            face_candidate.append((window, x, y, winW, winH, resized))
            
            
        
        #cv2.rectangle(resized, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        #cv2.imshow("Window", resized)
        #cv2.waitKey(1)
        #time.sleep(0.03)


#cv2.destroyAllWindows()

face_candidate = nms.non_maximum_suppression(face_candidate)



face_candidate24 = []
for window, x, y, winW, winH, resized in face_candidate:
    nonface, face = predict.predict_24Net(window, 0)
    if(face > 0.5):
        x, y, winW, winH, resized = calibration.pred_calibNet24((x, y, winW, winH, resized), 0)
        face_candidate24.append((window, x, y, winW, winH, resized))

face_candidate24 = nms.non_maximum_suppresion(face_candidate24)

face_candidate48 = []
for window, x, y, winW, winH, resized in face_candidate24:
    nonface, face = predict.predict_48Net(window, 0)
    if(face > 0.5):
        x, y, winW, winH, resized = calibration.pred_calibNet48((x, y, winW, winH, resized), 0)
        face_candidate48.append((window, x, y, winW, winH, resized))

face_candidate48 = nms.non_maximum_suppression_global(face_candidate48)



# print face_candidate

for face in face_candidate48:
    x, y, w, h, resized_image = face
    cv2.rectangle(resized_image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


