# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:14:01 2015

@author: yy374
"""

def non_maximum_suppression(face_candidate, max_list):
    new_face_candidate = [[]]
    for i in range(len(max_list)):    
        scale, clevel, xd, yd, wd, hd, resized_image = max_list[i]
        #print scale
        #print new_face_candidate[scale]
        #print "face_first: {}, {}, {}, {}, {}".format(xd, yd, wd, hd, resized_image.shape)
        new_face_candidate[scale].append((xd,yd,wd,hd,resized_image))
        for s in range(len(face_candidate)):
            if s != scale: continue
            for face in face_candidate[s]:
                x, y, w, h, resized = face
                ratio = IoU((xd,yd,wd,hd),(x,y,w,h))
                #print ratio, resized.shape
                if ratio < 0.3:
                    new_face_candidate[scale].append(face)
        new_face_candidate.append([])
    
    return new_face_candidate
        

def IoU(face_default, face):
    xd, yd, wd, hd = face_default
    #print "face_default: {}".format(face_default)
    x, y, w, h = face
    #print "face: {}".format(face)
    area_def = wd*hd
    if area_def == 0: return 1
    if y < yd:
        if x < xd:
            if h-(yd-y) < 0 or w-(xd-x) < 0:
                return 0.0
            area = (h-(yd-y))*(w-(xd-x))
            return area * 1.0 / area_def
        else:
            if h-(yd-y) < 0 or wd-(x-xd) < 0:
                return 0.0
            area = (h-(yd-y))*(wd-(x-xd))
            return area * 1.0 / area_def
    else:
        if x < xd:
            if hd-(y-yd) < 0 or w-(xd-x) < 0:
                return 0.0
            area = (hd-(y-yd))*(w-(xd-x))
            return area * 1.0 / area_def
        else:
            if hd-(y-yd) < 0 or wd-(x-xd) < 0:
                return 0.0
            area = (hd-(y-yd))*(wd-(x-xd))
            return area * 1.0 / area_def           

