import os

import numpy as np
import cv2

class Loader(object):
    '''Wrapper class to call appropriate Loader'''
    def __init__(self,path):
        e = os.path.splitext(path)[1]
        if e=='.avi':
            self.main = Loader_avi(path)
        elif e=='.mov':
            self.main = Loader_mov(path)
        self.cap = self.main.cap
        self.framenum = self.main.framenum
        self.frames = self.main.frames

    def getframe(self,fpos):
        '''Convert to Cartesian (flip rows) and return YXC'''
        frame =  self.main.getframe(fpos)
        frame = np.flip(frame,axis=0)
        return frame

class Loader_avi(object):
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        self.framenum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) -3 #due to bug
        self.frames = [self.cap.read()[1] for i in range(self.framenum)]
            
    def getframe(self,fpos):
        if fpos<0:
            return None
        elif fpos>=self.framenum:
            return None
        return self.frames[fpos]

class Loader_mov(object):
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        self.framenum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) -17 #due to bug
        self.frames = [self.cap.read()[1] for i in range(self.framenum)]

    def getframe(self,fpos):
        if fpos<0:
            return None
        elif fpos>=self.framenum:
            return None
        return self.frames[fpos]