import numpy as np

from PyQt5.QtWidgets import (QApplication, 
QPushButton,QLineEdit,QVBoxLayout)
from PyQt5.QtCore import pyqtSignal


import os,sys
sys.path.append(os.pardir)
from guitools import ImageBaseKeyControl,ViewControlBase
from visualize import DrawTextFixedPos
from movieimporter import Loader


class GravityFitBase(ImageBaseKeyControl):
    def __init__(self,parent=None):
        super().__init__(parent)
    def initUI(self):
        super().initUI()
        self.l1 = QVBoxLayout()
        self.l0.setStretch(0,3)
        self.l0.addLayout(self.l1,1)

        self.tb1 = QLineEdit()
        self.lb1 = QLineEdit('"first,last"')
        self.lb1.setReadOnly(True)
        self.button1 = QPushButton('enter')
        self.lb2 = QLineEdit()
        self.lb2.setReadOnly(True)
        self.button2 = QPushButton('finish')
        self.l1.addWidget(self.tb1)
        self.l1.addWidget(self.lb1)
        self.l1.addWidget(self.button1)
        self.l1.addWidget(self.lb2)
        self.l1.addWidget(self.button2)

        self.frametxt = DrawTextFixedPos(self.pli)

class GravityFitWidget(GravityFitBase):
    FrameRange = pyqtSignal(int,int)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.button1.clicked.connect(self.enter)
    def enter(self):
        txt = self.tb1.text()
        txtsplit = txt.split(',')
        if len(txtsplit) != 2:
            return
        t1,t2 = txtsplit
        if not t1.isdigit():
            return
        if not t2.isdigit():
            return
        first = int(t1)
        last = int(t2)
        self.FrameRange.emit(first,last)
    def showframe(self,fpos):
        self.frametxt.draw(f'frame: {fpos}')
    def showresult(self,gx,gy,gnorm):
        self.lb2.setText(f'gx:{gx:.3}, gy:{gy:.3}')

class GravityFitControl(ViewControlBase):
    def __init__(self,loader:Loader,pos:np.ndarray):
        super().__init__()
        self.fpos = 0
        self.ld = loader
        self.pos = pos
        self.calc = GravityEstimation(self.pos)
        self.window = GravityFitWidget()
        self.window.KeyPressed.connect(self.keyinterp)
        self.window.FrameRange.connect(self.run_calculation)
        self.change_fpos(self.fpos)
    def get_g(self):
        # (gx,gy,gnorm)
        return self.calc.get()
    def get_window(self):
        return self.window
    def get_frames(self):
        first,last = self.calc.get_frames()
        return first,last
    def finish_signal(self):
        return self.window.button2.clicked
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        self.window.setcvimage(self.ld.getframe(self.fpos))
        self.window.showframe(self.fpos)
        self.window.blockSignals(False)
    def run_calculation(self,first,last):
        self.calc.calc(first,last)
        self.window.showresult(*self.calc.get())


class GravityEstimation():
    def __init__(self,dpos):
        self.dpos = dpos
        self.dead = np.all(dpos==0,axis=1)

    def calc(self,first,last):
        self.first = first
        self.last = last
        dpos = self.dpos[first:last,:]
        dead = self.dead[first:last]
        frames = np.arange(first,last,dtype=int)
        dposx = np.delete(dpos[:,0],dead)
        dposy = np.delete(dpos[:,1],dead)
        frames = np.delete(frames,dead)

        self.xcoef,self.xres = self.gravity_fit(frames, dposx)
        self.ycoef,self.yres = self.gravity_fit(frames, dposy)
        self.accx = 2*self.xcoef[0]
        self.accy = 2*self.ycoef[0]
        self.gnorm = (self.accx**2 + self.accy**2)**0.5
    
    def get(self):
        ''' pixel/frame^2 '''
        return self.accx, self.accy, self.gnorm

    def get_frames(self):
        return self.first, self.last
    def gravity_fit(self,x,y):
        coef = np.polyfit(x,y,deg=2)
        res = y - np.polyval(coef,x)
        return coef,res

class GravityTest():
    '''test class'''
    def __init__(self):
        from analyzer import Results
        self.res = Results('../test/pro1')
        self.res.load()
        self.flpath = '../test/td_fall.mov'
        self.flld = Loader(self.flpath)
        self.df = self.res.f_oned.get()
        self.fit = GravityFitControl(self.flld,self.df[['d0c_x','d0c_y']].values)
        self.fit.get_window().show()
        self.fit.finish_signal().connect(self.a)
    def a(self):
        print(self.fit.get_g(),self.fit.get_frames())

def test():
    app = QApplication([])
    t = GravityTest()
    app.exec_()

if __name__=='__main__':
    test()