import sys,os
import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass,asdict,field

from PyQt5.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QPushButton
from PyQt5.QtCore import pyqtSignal,QTimer

sys.path.append(os.pardir)
from guitools import ConfigEditor,ImageBaseWidget,ViewControlBase
from visualize import DrawCircle


class CircleFitBase(QWidget):
    KeyPressed = pyqtSignal(int)
    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)
    def initUI(self):
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l1 = QVBoxLayout()

        self.l0.addLayout(self.l1,3)

        self.l2 = QVBoxLayout()
        self.l0.addLayout(self.l2,1)
        self.configedit = ConfigEditor(self)
        self.l2.addWidget(self.configedit)

class CircleFitWidget(CircleFitBase):
    def __init__(self,dianum,parent=None):
        super().__init__(parent)
        self.imwids = [ImageBaseWidget(self) for i in range(dianum)]
        self.circles = []
        for im in self.imwids:
            self.l1.addWidget(im)
            self.circles.append(DrawCircle(im.pli))
    def initUI(self):
        super().initUI()
        self.finish_button=QPushButton('finish',self)
        self.l2.addWidget(self.finish_button)
    def drawcircle(self,ix,cent,rad):
        self.circles[ix].draw(cent,rad)
    def setdict(self,d):
        self.configedit.setdict(d)

class CircleFitControl(ViewControlBase):
    def __init__(self,loader,boxdf):
        super().__init__()
        self.ld = loader
        self.boxdf = boxdf
        self.calc = CircleFit(self.ld,self.boxdf)
        self.calc.calc()
        self.first,self.last = self.calc.get_first_last()
        self.dianum = self.calc.get_dianum()
        self.window = CircleFitWidget(self.dianum)
        ini_d = asdict(self.calc.getconfig())
        self.window.setdict(ini_d)
        self.fpos=self.first
        self.change_fpos(self.fpos)

        self.window.KeyPressed.connect(self.keyinterp)
        self.window.configedit.UpdatedConfig.connect(self.config_update)

    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.finish_button.clicked
    def get_df(self):
        return self.calc.getdf()
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.first,self.last):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        crops,cents,rads = self.calc.get_crops_circles(self.fpos)
        for i,(crop,cent,rad) in enumerate(zip(crops,cents,rads)):
            if crop is None:
                continue
            self.window.imwids[i].setcvimage(crop)
            self.window.drawcircle(i,cent,rad)
        self.window.blockSignals(False)
        
    def config_update(self,d):
        new_c = CircleCnfg(**d)
        self.calc.setconfig(new_c)
        self.calc.calc()
        self.change_fpos(self.fpos)



@dataclass
class CircleCnfg:
    box_dilation: float = 0.6
    grad_p1: float = 1.0
    grad_p2: float = 1.0
    hough_p1: float = 300.0
    hough_p2: float = 1.0
    minrad: int = 5
    maxrad: int = 10

class CircleFit(object):
    def __init__(self,loader,box,masked=None):
        self.ld = loader
        self.masked = masked
        for i in range(99):
            if not np.isin(['d'+str(i)+'_x'],box.columns):
                break
            self.dianum=i+1

        basename = ['x','y','w','h']
        self.diabox = [None for i in range(self.dianum)]
        for i in range(self.dianum):
            name = ['d'+str(i)+'_'+n for n in basename]
            self.diabox[i] = box[name].values.astype(int)
        self.diacircle = [np.zeros((self.ld.framenum,3)) 
            for i in range(self.dianum)]
        
        self.c = CircleCnfg()
        self.bgskip = True
    
    def getdf(self):
        df = [None for i in range(self.dianum)]
        for i in range(self.dianum):
            df[i] = pd.DataFrame(self.diacircle[i],
                columns=['d'+str(i)+'c_x','d'+str(i)+'c_y','d'+str(i)+'c_r'])
        df = pd.concat(df,axis=1)
        return df
    def getconfig(self):
        return self.c
    def setconfig(self,cnfg:CircleCnfg):
        self.c = cnfg
    def get_dianum(self):
        return self.dianum
    def use_raw(self,skip=True):
        if skip:
            self.bgskip=True
        else:
            self.bgskip=False
    def calc(self):
        if self.bgskip:
            self._diafit_noBG()
            return
        for fp in range(0,self.ld.framenum):
            frame = self.masked[fp]
            circles = self._circle(frame,self.diabox[fp,:])
            if circles is not None:
                if circles.ndim==2:
                    i=0
                    cent = (circles[i,0],circles[i,1])
                    rad = circles[i,2]
                    if rad==0:
                        continue
                    self.diacircle[fp,0:2]=cent
                    self.diacircle[fp,2] = rad

    def _diafit_noBG(self):
        for j in range(self.dianum):
            for fp in range(0,self.ld.framenum):
                frame = self.ld.getframe(fp)
                circles = self._circle(frame,self.diabox[j][fp,:])
                if circles is None:
                    continue
                if circles.ndim!=2:
                    continue
                i=0
                cent = (circles[i,0],circles[i,1])
                rad = circles[i,2]
                if rad==0:
                    continue
                self.diacircle[j][fp,0:2]=cent
                self.diacircle[j][fp,2] = rad

    def get_first_last(self):
        first = self.ld.framenum
        last = 0
        for arr in self.diabox:
            ix = np.nonzero(np.any(arr!=0,axis=1))[0]
            f = ix.min()
            if first>f:
                first=f
            l = ix.max()+1
            if last<l:
                last = l
        return first,last

    def get_crops_circles(self,fpos):
        frame = self.ld.getframe(fpos)
        rad = [None for i in range(self.dianum)]
        cent = [None for i in range(self.dianum)]
        crop = [None for i in range(self.dianum)]
        for i in range(self.dianum):
            cent[i] = self.diacircle[i][fpos,0:2]
            rad[i] = self.diacircle[i][fpos,2]
            crop[i],l,u,_,_ = self._getroi(frame,self.diabox[i][fpos,:])
            cent[i] = [cent[i][0]-l,cent[i][1]-u]
        return crop,cent,rad
        

    def _circle(self,frame,box):
        roi,x,y,w,h = self._getroi(frame,box)
        if roi is None:
            return None
        gr = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gr,cv2.HOUGH_GRADIENT,
            self.c.grad_p1, self.c.grad_p2, (w+h)/8, self.c.hough_p1, self.c.hough_p2,
            minRadius = self.c.minrad, maxRadius = self.c.maxrad)
        if circles.ndim==3:
            circles = circles[0]
        elif circles.ndim==2:
            return None
        circles[:,0] = circles[:,0]+x
        circles[:,1] = circles[:,1]+y
        return circles
    
    def _getroi(self,frame,box):
        x,y,w,h = box
        if w==0 or h==0:
            return None, None, None, None, None
        cent = [int(x+w/2), int(y+h/2)]
        wid = int((w/2)*self.c.box_dilation)
        hig = int((h/2)*self.c.box_dilation)
        l,r,u,d = [cent[0]-wid, cent[0]+wid, cent[1]-hig, cent[1]+hig]
        if l<0:
            l = 0
        if u<0:
            u = 0
        if r>frame.shape[1]:
            r = frame.shape[1]
        if d>frame.shape[0]:
            d = frame.shape[0]
        
        roi = frame[u:d,l:r,:]
        return roi, l, u, w, h



class TestCircle():
    '''test class'''
    def __init__(self):
        from movieimporter import Loader
        from analyzer import Results
        path = '../test/pro2'
        res = Results(path)
        res.load()
        impath = '../test/td2.mov'
        ld = Loader(impath)
        df = res.oned.get()
        self.circle = CircleFitControl(ld,df)
        self.circle.get_window().show()
        self.circle.finish_signal().connect(self.aa)
    def aa(self):
        print('yay')
        df = self.circle.get_df()

def test():
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    t = TestCircle()
    app.exec_()

if __name__=='__main__':
    test()