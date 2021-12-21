import cv2
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.pardir)
from guitools import ROItool,ViewControlBase,ImageBaseKeyControl,RoiSelector
from visualize import NewDrawing
from PyQt5.QtCore import  Qt, pyqtSignal,QTimer
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QLineEdit



class TrackerWidget(ImageBaseKeyControl):
    ROIselected = pyqtSignal(tuple)
    TextEdited = pyqtSignal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.drawing = NewDrawing(self.pli)

        self.roi = RoiSelector(self.pli)

        self.l0.setStretch(0,4)
        self.l1 = QVBoxLayout()
        self.l0.addLayout(self.l1,1)

        self.label = QLabel(self)
        self.l1.addWidget(self.label)

        self.button = QPushButton('determine ROI')
        self.l1.addWidget(self.button)

        self.txtbox = QLineEdit('l',self)
        self.l1.addWidget(self.txtbox)
        self.tb_button=QPushButton('determine target (l,r,d0,d1,d2,...)')
        self.l1.addWidget(self.tb_button)

        self.fin_button = QPushButton('Finish')
        self.l1.addWidget(self.fin_button)

        self.button.clicked.connect(self._ROIselected)
        self.tb_button.clicked.connect(self._txtedited)

    def _ROIselected(self):
        self.ROIselected.emit(self.roi.get_ROI())

    def _txtedited(self):
        txt = self.txtbox.text()
        self.TextEdited.emit(txt)

    def add_rec(self,key,recs):
        self.drawing.set_rectangle(key,recs)

    def change_color(self,key):
        sty = self.drawing.getsty()
        for unitkey,val in sty.items():
            if unitkey[-4:]!='_rec':
                continue
            if unitkey.split('_')[0]==key:
                val['pen']['color']='g'
            else:
                val['pen']['color']='c'
        self.drawing.setsty(sty)



class OldTrackerWidget(ROItool):
    '''depreciated'''
    def __init__(self,parent=None):
        super().__init__(parent)
        self.drawing = NewDrawing(self.pli)
        # self.rectangles = {}
    def setcvimage(self,im):
        nim = np.swapaxes(im,0,1)
        nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
        self.imi.setImage(nim)
    def add_rec(self,key,recs):
        self.drawing.set_rectangle(key,recs)
    def change_color(self,key):
        sty = self.drawing.getsty()
        for unitkey,val in sty.items():
            if unitkey[-4:]!='_rec':
                continue
            if unitkey.split('_')[0]==key:
                val['pen']['color']='g'
            else:
                val['pen']['color']='c'
        self.drawing.setsty(sty)

class TrackerMain(ViewControlBase):
    def __init__(self,masked):
        super().__init__()
        self.window = TrackerWidget()
        self.calc = ObjectTracker(masked)
        self.masked = masked
        self.framenum = len(masked)
        self.fpos = 0
        self.set_target('l')
        self.change_fpos(1)

        self.window.ROIselected.connect(self.run_track)
        self.window.TextEdited.connect(self.set_target)
        self.window.KeyPressed.connect(self.keyinterp)
    def finish_signal(self):
        return self.window.fin_button.clicked
    def get_window(self):
        return self.window
    def get_df(self):
        return self.calc.getdf()
    def get_dianum(self):
        return self.calc.get_dianum()
    def change_fpos(self,new_fpos):
        if new_fpos not in range(self.framenum):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        self.window.setcvimage(self.masked[new_fpos])
        self.window.drawing.update(self.fpos)
        self.window.blockSignals(False)
    def run_track(self,rec):
        self.calc.track(self.target,self.fpos,self.framenum,rec)
        self.set_rec(self.target)
        self.set_target(self.target)
        self.change_fpos(self.fpos)
    def set_rec(self,key):
        recs = self.calc.getarr(key)
        self.window.add_rec(key,recs)
    def set_target(self,txt):
        self.target = txt
        self.window.label.setText(f'target: {txt}')
        self.window.change_color(txt)
        self.change_fpos(self.fpos)


class ObjectTracker(object):
    def __init__(self,masked):
        self.masked = masked
        self.framenum = len(self.masked)

        self.diabox = [np.zeros((self.framenum,4))]
        self.lbox = np.zeros((self.framenum,4))
        self.rbox = np.zeros((self.framenum,4))
    def getdf(self):
        basename = ['_x','_y','_w','_h']
        df = [None for i in range(len(self.diabox)+2)]
        name = ['l'+n for n in basename]
        df[0] = pd.DataFrame(self.lbox,columns=name)
        name = ['r'+n for n in basename]
        df[1] = pd.DataFrame(self.rbox,columns=name)
        for i in range(len(self.diabox)):
            name = ['d'+str(i)+n for n in basename]
            df[i+2] = pd.DataFrame(self.diabox[i],columns=name)
        outdf = pd.concat((df),1)
        return outdf
    def get_dianum(self):
        return len(self.diabox)
    def get(self,fp):
        l = self.lbox[fp,:]
        r = self.rbox[fp,:]
        d = [dbox[fp,:] for dbox in self.diabox]
        return l,r,d
    def getarr(self,key):
        if key=='l':
            return self.lbox
        if key=='r':
            return self.rbox
        if key[0]=='d':
            ix = int(key[1:])
            return self.diabox[ix]
    def track(self,key,spos,epos,initBB):
        if key=='l':
            self.lhand(spos,epos,initBB)
        elif key=='r':
            self.rhand(spos,epos,initBB)
        elif key[0]=='d':
            if not key[1:].isdigit():
                return
            ix = int(key[1:])
            self.dia(ix,spos,epos,initBB)
    def dia(self,ix,spos,epos,initBB):
        if ix not in range(len(self.diabox)):
            l = ix+1-len(self.diabox)
            add = [np.zeros((self.framenum,4)) for i in range(l)]
            self.diabox += add
        self.diabox[ix][spos:epos,:] = self._tracking(spos,epos,initBB)[spos:epos,:]
    def lhand(self,spos,epos,initBB):
        self.lbox[spos:epos,:] = self._tracking(spos,epos,initBB)[spos:epos,:]
    def rhand(self,spos,epos,initBB):
        self.rbox[spos:epos,:] = self._tracking(spos,epos,initBB)[spos:epos,:]

    def _tracking(self,spos,epos,initBB):
        box = np.zeros((self.framenum,4))
        frame = self.masked[spos]
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, initBB)
        box[spos,:] = [int(v) for v in initBB]
        for fp in range(spos+1,epos):
            # print(f'tracking {fp}')
            frame = self.masked[fp]
            success, b = tracker.update(frame)
            if not success:
                break
            x, y, w, h = [int(v) for v in b]
            box[fp,:] = (x,y,w,h)
        return box





class TestTracker():
    '''test class'''
    def __init__(self):
        from movieimporter import Loader
        from BGcorrection import BackgroundCorrection, MaskMaker
        impath = '../test/td1.mov'
        ld = Loader(impath)
        bg = BackgroundCorrection(ld)
        bg.calc()
        mm = MaskMaker(ld,bg.get())
        mm.calc()
        self.masked = mm.get()
        self.tracker = TrackerMain(self.masked)
        self.tracker.get_window().show()
        self.tracker.finish_signal().connect(self.aa)
    def aa(self):
        print('yay')
        # df = self.tracker.get_df()
        # df.to_csv('./test7/testing_2.csv',index_label=True)

def test():
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    tt = TestTracker()
    app.exec_()

if __name__=='__main__':
    test()