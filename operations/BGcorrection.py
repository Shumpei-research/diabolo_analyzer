import cv2
import numpy as np
from dataclasses import dataclass, asdict, field

import sys,os
sys.path.append(os.pardir)
from movieimporter import Loader

class BackgroundCorrection(object):
    def __init__(self,loader:Loader):
        self.ld = loader
        self.fgmask = [None for i in range(self.ld.framenum)]

    def calc(self):
        backSub = cv2.createBackgroundSubtractorMOG2()
        # for initial frames
        for fp in range(0,30):
            # print(f'bg: {fp}')
            frame = self.ld.getframe(fp)
            self.fgmask[fp] = backSub.apply(frame)
        for fp in range(0,self.ld.framenum):
            # print(f'bg: {fp}')
            frame = self.ld.getframe(fp)
            self.fgmask[fp] = backSub.apply(frame).astype(np.uint8)
    
    def save(self,path_fgmask):
        np.save(path_fgmask,np.stack(self.fgmask,axis=0))
    
    def get(self):
        return self.fgmask

@dataclass
class MaskCnfg:
    bgcolor: list = field(default_factory=lambda: [0,0,0])

class MaskMaker(object):
    def __init__(self,loader,fgmask):
        self.ld = loader
        self.fgmask = fgmask
        self.c = MaskCnfg()
        self.masked = [None for i in range(self.ld.framenum)]
    def calc(self):
        for fp in range(0,self.ld.framenum):
            # print(f'masking: {fp}')
            frame = self.ld.getframe(fp)
            self.masked[fp] = np.where(np.expand_dims(self.fgmask[fp],2),
                frame,
                np.expand_dims(self.c.bgcolor,(0,1))).astype(np.uint8)
    def setcnfg(self,c:MaskCnfg):
        self.c = c
    def get(self):
        return self.masked