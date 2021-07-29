from operator import index
import os
import json
import copy
from PyQt5.QtCore import pyqtSignal
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from dataclasses import dataclass, asdict, field

class Loader(object):
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
        return self.main.getframe(fpos)

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


from PyQt5.QtWidgets import QHBoxLayout, QLabel, QTabWidget, QVBoxLayout
from guitools import EditorAndView, ImageBaseKeyControl, ROItool, StickPositionEditorBase
from visualize import DrawRectangle, NewDrawing

class TrackerWidget(ROItool):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.rectangles = {}
    def setcvimage(self,im):
        nim = np.swapaxes(im,0,1)
        nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
        self.imi.setImage(nim)
    def show_rec(self,rec,key):
        if key not in self.rectangles.keys():
            self._add_rec(key)
        self.rectangles[key].setvis(True)
        self.rectangles[key].draw(*rec)
    def remove_rec(self,key):
        if key in self.rectangles.keys():
            self.rectangles[key].setvis(False)
    def _add_rec(self,key):
        self.rectangles[key]=DrawRectangle(self.pli)
        self.remove_rec(key)
    def change_color(self,key):
        for k,item in self.rectangles.items():
            sty = item.getsty()
            if k==key:
                sty['pen']['color']='g'
            else:
                sty['pen']['color']='c'
            item.setsty(sty)

from guitools import ViewControlBase

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
        self._show_roi()
        self.window.blockSignals(False)
    def run_track(self,rec):
        self.calc.track(self.target,self.fpos,self.framenum,rec)
        self.change_fpos(self.fpos)
        self.set_target(self.target)
    def set_target(self,txt):
        self.target = txt
        self.window.label.setText(f'target: {txt}')
        self.window.change_color(txt)
        self.change_fpos(self.fpos)
    def _show_roi(self):
        def show(rec,key):
            if np.all(rec==0):
                self.window.remove_rec(key)
            else:
                self.window.show_rec(rec,key)
        l,r,d = self.calc.get(self.fpos)
        show(l,'l')
        show(r,'r')
        for i,rec in enumerate(d):
            key = 'd'+str(i)
            show(rec,key)

class TestTracker():
    '''test class'''
    def __init__(self):
        impath = './test7/td2.mov'
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
        df = self.tracker.get_df()
        df.to_csv('./test7/testing_2.csv',index_label=True)


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



from guitools import ImageBaseWidget,CircleFitBase
from PyQt5.QtWidgets import QPushButton
from visualize import DrawCircle

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

class TestCircle():
    '''test class'''
    def __init__(self):
        impath = './test7/td2.mov'
        ld = Loader(impath)
        df = pd.read_csv('./test7/testing_2.csv',index_col=0)
        self.circle = CircleFitControl(ld,df)
        self.circle.get_window().show()
        self.circle.finish_signal().connect(self.aa)
    def aa(self):
        print('yay')
        df = self.circle.get_df()
        df.to_csv('./test7/testing_2_circle.csv',index_label=True)


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
        



from guitools import StickFinderBase
from visualize import DrawPos
import pyqtgraph as pg
from PyQt5.QtGui import QPainter

class StickPosition(object):
    def __init__(self,framenum):
        '''
        leftstick: column 0, rightstick: column 1
        left:1 right:2 flying:3 absent:0
        '''
        self.loadplace(np.zeros((framenum,2),dtype=int))
    def _chpos(self):
        d = self.place[1:,:] - self.place[:-1,:]
        ch = np.any(d!=0,axis=1)
        chpos = np.nonzero(ch)[0]+1
        chpos = np.insert(chpos,0,0) # first frame
        return chpos
    def loadplace(self,place):
        self.place = place
        self.chpos = self._chpos()
        self.state = np.array([self.place[p,:] for p in self.chpos])
    def loadchanges(self,chpos,state):
        self.chpos = np.array(chpos,dtype=int)
        self.state = np.array(state,dtype=int)
        self._setplace()
    def loadchanges_array(self,arr):
        chpos = arr[:,0]
        state = arr[:,1:3]
        self.loadchanges(chpos,state)
    def _setplace(self):
        if self.chpos.size==1:
            self.place[self.chpos[0]:,:] = np.expand_dims(self.state[0],axis=0)
            return
        for n,(i,j) in enumerate(zip(self.chpos[:-1],self.chpos[1:])):
            self.place[i:j,:] = np.expand_dims(self.state[n],axis=0)
        self.place[self.chpos[-1]:,:] = np.expand_dims(self.state[-1],axis=0)
    def get(self):
        return self.place,self.chpos,self.state
    def get_array(self):
        return np.concatenate((np.expand_dims(self.chpos,axis=1),self.state),axis=1)
    def getiter(self):
        spos = self.chpos
        epos = np.append(self.chpos[1:],len(self.place))
        st = [self.state[i,:] for i in range(self.state.shape[0])]
        return zip(spos,epos,st)
    def where(self,l,r):
        lp = np.isin(self.state[:,0],l)
        rp = np.isin(self.state[:,1],r)
        ix = np.logical_and(lp,rp)
        ix2 = np.insert(ix[:-1],0,False)
        posstart = self.chpos[ix]
        posend = self.chpos[ix2]
        if ix[-1]==True:
            posend = np.append(posend,self.place.shape[0])
        return posstart,posend
    def swap_frame(self):
        normal_s1,normal_e = self.where([2],[1,3])
        normal_s2,normal_e = self.where([2,3],[1])
        normal_s = np.unique(np.concatenate((normal_s1,normal_s2)))
        if normal_s.size==1:
            return np.array([])
        return normal_s[1:]



from guitools import EditorAndView
from visualize import DrawTextFixedPos

class StickPositionEditorWidget(EditorAndView):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.editor
    def initUI(self):
        super().initUI()
        self.frametxt = DrawTextFixedPos(self.viewer.pli)
        self.finishbutton = QPushButton('finish')
        self.l1.addWidget(self.finishbutton)
    def drawframe(self,fpos):
        self.frametxt.draw(f'frame: {fpos}')
    def setcvimage(self,im):
        self.viewer.setcvimage(im)
    def set_array(self,arr):
        self.editor.from_array(arr)
    def get_array(self):
        return self.editor.to_array()

class StickPositionEditorControl(ViewControlBase):
    def __init__(self,loader:Loader,stickpos:StickPosition=None):
        super().__init__()
        self.ld = loader
        self.window = StickPositionEditorWidget()
        self.fpos=0
        self.stickpos = StickPosition(self.ld.framenum)
        self.stickpos.loadchanges([0],[1,2])
        if stickpos is not None:
            self.stickpos = stickpos
        self.change_fpos(self.fpos)
        self.window.KeyPressed.connect(self.keyinterp)
    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.finishbutton.clicked
    def get_stickpos(self):
        currentarr = self.window.get_array()
        self.stickpos.loadchanges_array(currentarr)
        return self.stickpos
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos=new_fpos
        self.window.drawframe(self.fpos)
        self.window.setcvimage(self.ld.getframe(self.fpos))
        self.window.blockSignals(False)


class StickFinderWidget(StickFinderBase):
    def __init__(self,parent=None):
        super().__init__(parent)
    def initUI(self):
        super().initUI()
        self.positions = [DrawPos(self.imwids[i].pli) for i in range(2)]
        self.hsvpositions = [DrawPos(self.hsvwids[i].pli) for i in range(2)]
    def drawpos(self,ix,pos):
        self.positions[ix].draw(pos)
        self.hsvpositions[ix].draw(pos)
    def setdict(self,d):
        self.configedit.setdict(d)

class StickFinderControl(ViewControlBase):
    def __init__(self,loader,boxdf,stickpos):
        super().__init__()
        self.ld = loader
        self.boxdf = boxdf
        self.stickpos=stickpos
        self.calc = StickFinder(self.ld,self.boxdf,self.stickpos)
        self.calc.main()
        self.first,self.last = self.calc.get_first_last()
        self.window = StickFinderWidget()
        self.window.show()
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
        return self.calc.get_df()
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.first,self.last):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        lpos,lim,lhsv,lreg,rpos,rim,rhsv,rreg = self.calc.get(self.fpos)
        if lim is not None:
            self.window.imwids[0].setcvimage(lim)
            self.window.hsvwids[0].setcvimage(lhsv)
            self.window.imwids[0].setmask(lreg)
            self.window.hsvwids[0].setmask(lreg)
            self.window.drawpos(0,lpos)
        if rim is not None:
            self.window.imwids[1].setcvimage(rim)
            self.window.hsvwids[1].setcvimage(rhsv)
            self.window.imwids[1].setmask(rreg)
            self.window.hsvwids[1].setmask(rreg)
            self.window.drawpos(1,rpos)
        self.window.blockSignals(False)
        
    def config_update(self,d):
        new_c = StickFinderCnfg(**d)
        self.calc.setcnfg(new_c)
        self.calc.main()
        self.change_fpos(self.fpos)




from PyQt5.QtWidgets import QTabWidget

class TestStick(QTabWidget):
    '''test class'''
    def __init__(self,parent=None):
        super().__init__(parent)
        impath = './test7/td2.mov'
        self.ld = Loader(impath)
        self.df = pd.read_csv('./test7/testing_2.csv',index_col=0)
        self.stickpos = StickPosition(self.ld.framenum)
        self.posedit = StickPositionEditorControl(self.ld)
        self.addTab(self.posedit.get_window(),'aa')
        self.posedit.finish_signal().connect(self.next)

    def next(self):
        self.stickpos = self.posedit.get_stickpos()
        self.stick = StickFinderControl(self.ld,self.df,self.stickpos)
        self.addTab(self.stick.get_window(),'aiai')
        self.setTabEnabled(0,False)
        self.stick.finish_signal().connect(self.aa)
    
    def aa(self):
        print('yay')
        print(self.stick.get_df().iloc[600:610,:])
        res = self.stick.get_df()
        newdf = pd.concat((self.df,res),axis=1)
        # newdf.to_csv('./test7/testing_3.csv')




@dataclass
class StickFinderCnfg:
    hcen: int = 170
    hwid: int = 10
    smin: int = 70
    smax: int = 255
    vmin: int = 70
    vmax: int = 255
    sticklen: int=100
    dilation: int=5
    minsize: int=120
    searchlen: int=100
    dilation_fly: int=7
    minsize_fly: int=100


class StickFinder(object):
    '''
    where sticks belong (left/right/flying) will be provided
    detect regions
    clean up regions
    if left and right:
        see if each region belongs to left or right
    if flying:
        'flying' region will be tracked
        'flying' regions are ignored in the following steps
    if left(/right) and flying:
        operate on remaining regions
    '''
    def __init__(self,loader,box,stickpos):
        self.ld = loader
        self.box = box
        self.spos = stickpos

        self.c = StickFinderCnfg()

        self.lbox = self.box[['l_x','l_y','l_w','l_h']].values
        self.rbox = self.box[['r_x','r_y','r_w','r_h']].values
        self.lc = self._getcenter(self.lbox)
        self.rc = self._getcenter(self.rbox)
        self.isoverlap = self._isoverlap(self.lc,self.rc,self.c.sticklen)

        self.lrec = [None for i in range(self.ld.framenum)]
        self.rrec = [None for i in range(self.ld.framenum)]
        self.lreg = [None for i in range(self.ld.framenum)]
        self.rreg = [None for i in range(self.ld.framenum)]
        self.ltip_x = [None for i in range(self.ld.framenum)]
        self.ltip_y = [None for i in range(self.ld.framenum)]
        self.rtip_x = [None for i in range(self.ld.framenum)]
        self.rtip_y = [None for i in range(self.ld.framenum)]

        l = self.c.sticklen
        self.dmap = self._distancemap(2*l,2*l)
    
    def setcnfg(self,c:StickFinderCnfg):
        self.c = c
    def getconfig(self):
        return self.c
    def get_first_last(self):
        first = self.ld.framenum
        last = 0
        for arr in (self.lbox,self.rbox):
            ix = np.nonzero(np.any(arr!=0,axis=1))[0]
            f = ix.min()
            if first>f:
                first=f
            l = ix.max()+1
            if last<l:
                last = l
        return first,last
    def get_df(self):
        recc = ['xmin','xmax','ymin','ymax']
        lrecc = ['l_sti_'+i for i in recc]
        rrecc = ['r_sti_'+i for i in recc]
        lrec = [[0,0,0,0] if i is None else i for i in self.lrec]
        rrec = [[0,0,0,0] if i is None else i for i in self.rrec]
        lrec = np.stack(lrec,axis=0)
        rrec = np.stack(rrec,axis=0)
        dfl = pd.DataFrame(lrec,columns=lrecc)
        dfr = pd.DataFrame(rrec,columns=rrecc)
        ltip_x = np.array(self.ltip_x)
        ltip_x = np.where(ltip_x==None,0,ltip_x)
        ltip_y = np.array(self.ltip_y)
        ltip_y = np.where(ltip_y==None,0,ltip_y)
        rtip_x = np.array(self.rtip_x)
        rtip_x = np.where(rtip_x==None,0,rtip_x)
        rtip_y = np.array(self.rtip_y)
        rtip_y = np.where(rtip_y==None,0,rtip_y)
        dftip = pd.DataFrame(np.stack([ltip_x,ltip_y,rtip_x,rtip_y],axis=1),
            columns=['ltip_x','ltip_y','rtip_x','rtip_y'])
        df = pd.concat((dfl,dfr,dftip),axis=1)
        return df
    def get(self,fpos):
        lrec = self.lrec[fpos]
        rrec = self.rrec[fpos]
        frame = self.ld.getframe(fpos)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        if lrec is not None:
            ltipx, ltipy = self.ltip_x[fpos],self.ltip_y[fpos]
            ltipx -=lrec[0]
            ltipy -=lrec[2]
            lframe = self._crop(frame,lrec)
            lhsv = self._crop(hsv,lrec)
            # lreg = cv2.cvtColor(self.lreg[fpos].astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)             
            lreg = self.lreg[fpos].astype(np.uint8)             
        else:
            ltipx = None
            ltipy = None
            lframe = None
            lhsv = None
            lreg = None
        if rrec is not None:
            rtipx, rtipy = self.rtip_x[fpos],self.rtip_y[fpos]
            rtipx -=rrec[0]
            rtipy -=rrec[2]
            rframe = self._crop(frame,rrec)
            rhsv = self._crop(hsv,rrec)
            # rreg = cv2.cvtColor(self.rreg[fpos].astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)             
            rreg = self.rreg[fpos].astype(np.uint8)             
        else:
            rtipx = None
            rtipy = None
            rframe = None
            rhsv = None
            rreg = None
        return (ltipx,ltipy),lframe,lhsv,lreg,(rtipx,rtipy),rframe,rhsv,rreg
    
    def main(self):
        print('calculating stick position')
        for s,e,state in self.spos.getiter():
            if np.all(state==[1,2]):
                # held
                self.both(s,e)
            elif np.all(state==[2,1]):
                # reverse held
                self.both(s,e)
                self._bothswap(s,e)
            elif state[0]==3 and state[1]==3:
                self.flying('l',s,e)
                self.flying('r',s,e)
            elif state[0]==3:
                # left flying
                self.flying('l',s,e)
                self.onehand('r',s,e)
            elif state[1]==3:
                # right flying
                self.flying('r',s,e)
                self.onehand('l',s,e)
        print('done')

    def flying(self,l_r,start,end):
        if l_r == 'l':
            tipx = copy.deepcopy(self.ltip_x[start-1])
            tipy = copy.deepcopy(self.ltip_y[start-1])
            tip = np.array([tipx,tipy])
            res = self.flying_main(start,end,tip)
            self.lreg[start:end] = res[0]
            self.lrec[start:end] = res[1]
            restip = res[2]
            self.ltip_x[start:end] = [r[0] for r in restip]
            self.ltip_y[start:end] = [r[1] for r in restip]
        if l_r == 'r':
            tipx = copy.deepcopy(self.rtip_x[start-1])
            tipy = copy.deepcopy(self.rtip_y[start-1])
            tip = np.array([tipx,tipy])
            res = self.flying_main(start,end,tip)
            self.rreg[start:end] = res[0]
            self.rrec[start:end] = res[1]
            restip = res[2]
            self.rtip_x[start:end] = [r[0] for r in restip]
            self.rtip_y[start:end] = [r[1] for r in restip]

    def flying_main(self,start,end,inipos):
        res_reg = [None for fp in range(end-start)]
        res_rec = [None for fp in range(end-start)]
        res_pos = [None for fp in range(end-start)]
        res_crop = [None for fp in range(end-start)]

        inivel = np.array([0,0]) #x,y
        for fp in range(start,end):
            frame = self.ld.getframe(fp)
            pos = inipos+inivel
            search_rec = self._getrectangle(frame.shape,pos,
                self.c.searchlen)
            crop = self._crop(frame,search_rec)
            reg,_ = self._colorfilter(crop)
            reg = self._cleanup(reg,self.c.dilation_fly,self.c.minsize_fly)
            intpos = pos - np.array([search_rec[0],search_rec[2]])
            reg, newintpos = self._closest(reg,intpos)
            newpos = newintpos+np.array([search_rec[0],search_rec[2]])
            inivel = newpos-inipos
            inipos = newpos

            res_reg[fp-start] = reg
            res_rec[fp-start] = search_rec
            res_pos[fp-start] = newpos
            res_crop[fp-start] = crop
        return res_reg, res_rec, res_pos
            
    def _closest(self,reg,pos):
        lab, lnum = ndi.label(reg)
        com = ndi.center_of_mass(reg,lab,np.arange(1,lnum+1))
        disp = np.array([np.array(regp) for regp in com])
        disp = disp - np.expand_dims([pos[1],pos[0]],axis=0)
        d = np.linalg.norm(disp,axis=1)
        ix = d.argmin()
        chosen = ix+1
        chosenreg = (lab==chosen)
        chosenpos = np.array([com[ix][1],com[ix][0]],dtype=int)
        return chosenreg,chosenpos

    def onehand(self,hand,start,end):
        if hand == 'l':
            for fp in range(start,end):
                frame = self.ld.getframe(fp)
                lrec = self._getrectangle(frame.shape,np.squeeze(self.lc[fp,:]),self.c.sticklen)
                lcrop = self._crop(frame,lrec)
                lreg,lhsv = self._colorfilter(lcrop)
                fly_reg = self.rreg[fp]
                fly_rec = self.rrec[fp]
                mask = self._movereg(fly_reg,fly_rec,lrec)
                lreg = np.logical_and(lreg,~mask)
                lreg = self._cleanup(lreg,self.c.dilation,self.c.minsize)
                self.ltip_x[fp], self.ltip_y[fp] = self._tip(lreg,lrec,self.lc[fp],self.dmap)
                self.lrec[fp] = lrec
                self.lreg[fp] = lreg

        if hand == 'r':
            for fp in range(start,end):
                frame = self.ld.getframe(fp)
                rrec = self._getrectangle(frame.shape,np.squeeze(self.rc[fp,:]),self.c.sticklen)
                rcrop = self._crop(frame,rrec)
                rreg,rhsv = self._colorfilter(rcrop)
                fly_reg = self.lreg[fp]
                fly_rec = self.lrec[fp]
                mask = self._movereg(fly_reg,fly_rec,rrec)
                rreg = np.logical_and(rreg,~mask)
                rreg = self._cleanup(rreg,self.c.dilation,self.c.minsize)
                self.rtip_x[fp], self.rtip_y[fp] = self._tip(rreg,rrec,self.rc[fp],self.dmap)
                self.rrec[fp] = rrec
                self.rreg[fp] = rreg

    def _movereg(self,region,ori_rec,des_rec):
        mergerec,ori_inrec,des_inrec = self._mergerectangel(
            ori_rec,des_rec)
        temp = np.zeros((mergerec[3]-mergerec[2],mergerec[1]-mergerec[0]),dtype=region.dtype)
        temp[ori_inrec[2]:ori_inrec[3],ori_inrec[0]:ori_inrec[1]] = region
        return temp[des_inrec[2]:des_inrec[3],des_inrec[0]:des_inrec[1]]

    def both(self,start,end):
        for fp in range(start,end):
            if np.all(self.lc[fp]==0) and np.all(self.rc[fp]==0):
                continue
            frame = self.ld.getframe(fp)
            lrec = self._getrectangle(frame.shape,np.squeeze(self.lc[fp,:]),self.c.sticklen)
            rrec = self._getrectangle(frame.shape,np.squeeze(self.rc[fp,:]),self.c.sticklen)
            
            lcrop = self._crop(frame,lrec)
            rcrop = self._crop(frame,rrec)
            if self.isoverlap[fp]:
                mergerec, linrec, rinrec = self._mergerectangel(lrec,rrec)
                mergecrop = self._crop(frame,mergerec)
                region,hsv = self._colorfilter(mergecrop)
                lhsv = self._crop(hsv,linrec)
                rhsv = self._crop(hsv,rinrec)
                cleaned = self._cleanup(region,self.c.dilation,self.c.minsize)
                lreg,rreg = self._separate(cleaned,mergerec,self.lc[fp,:],self.rc[fp,:],linrec,rinrec)
            else:
                lreg,lhsv = self._colorfilter(lcrop)
                rreg,rhsv = self._colorfilter(rcrop)
                lreg = self._cleanup(lreg,self.c.dilation,self.c.minsize)
                rreg = self._cleanup(rreg,self.c.dilation,self.c.minsize)
            
            self.lrec[fp] = lrec
            self.rrec[fp] = rrec
            self.lreg[fp] = lreg
            self.rreg[fp] = rreg

            self.ltip_x[fp], self.ltip_y[fp] = self._tip(lreg,lrec,self.lc[fp],self.dmap)
            self.rtip_x[fp], self.rtip_y[fp] = self._tip(rreg,rrec,self.rc[fp],self.dmap)

    def _bothswap(self,s,e):
        self._lrswap(self.lrec,self.rrec,s,e)
        self._lrswap(self.lreg,self.rreg,s,e)
        self._lrswap(self.ltip_x,self.rtip_x,s,e)
        self._lrswap(self.ltip_y,self.rtip_y,s,e)

    def _lrswap(self,l,r,start,end):
        temp = l[start:end]
        l[start:end] = r[start:end]
        r[start:end] = temp
        
    def _separate(self,reg,regrec,cent1,cent2,intrec1,intrec2):
        labeled,lnum = ndi.label(reg)
        regcent = ndi.center_of_mass(reg,labeled,np.arange(1,lnum+1))
        regcent = [[r[1],r[0]] for r in regcent]
        intcent1 = cent1 - [regrec[0],regrec[2]]
        intcent2 = cent2 - [regrec[0],regrec[2]]
        disp1 = np.array([r - intcent1 for r in regcent])
        disp2 = np.array([r - intcent2 for r in regcent])
        d1 = np.linalg.norm(disp1,axis=1)
        d2 = np.linalg.norm(disp2,axis=1)
        isfirst = d1<=d2
        label1 = np.arange(1,lnum+1)[isfirst]
        label2 = np.arange(1,lnum+1)[~isfirst]
        reg1 = np.isin(labeled,label1)
        reg2 = np.isin(labeled,label2)
        reg1 = reg1[intrec1[2]:intrec1[3],intrec1[0]:intrec1[1]]
        reg2 = reg2[intrec2[2]:intrec2[3],intrec2[0]:intrec2[1]]
        return reg1,reg2

    def _cleanup(self,region,d,minsize):
        newreg = ndi.binary_dilation(region,structure=np.ones((d,d)))
        smnewreg = self._filter_small(newreg,minsize)
        return smnewreg

    def _filter_small(self,region,minsize):
        connectivity = np.ones((3,3))
        lab, ln = ndi.label(region,structure=connectivity)
        un, counts = np.unique(lab,return_counts=True)
        label = un[counts>minsize]
        label = label[label!=0]
        newregion = np.isin(lab,label)
        return newregion
        
    def _colorfilter(self,im):
        ch = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        hrange = np.array(range(self.c.hcen-self.c.hwid , self.c.hcen+self.c.hwid+1))
        hrange = np.where(hrange<0,hrange+180,hrange)
        hrange = np.where(hrange>179,hrange-180,hrange)
        hreg = np.isin(ch[:,:,0],hrange)

        srange = [self.c.smin,self.c.smax]
        vrange = [self.c.vmin,self.c.vmax]
        sreg = np.logical_and(srange[0]<=ch[:,:,1],srange[1]>=ch[:,:,1])
        vreg = np.logical_and(vrange[0]<=ch[:,:,2],vrange[1]>=ch[:,:,2])
        
        region = np.where(np.all(np.stack((hreg,vreg,sreg),axis=0),axis=0),255,0).astype(np.uint8)
        return region,ch
        
    def _getcenter(self,box):
        x,y,w,h = [box[:,i] for i in range(4)]
        centint = np.stack([x+(w/2),y+(h/2)],axis=1).astype(int)
        return centint

    def _isoverlap(self,cent1,cent2,l):
        d = cent1-cent2
        isoverlap = np.all((np.abs(d) <= 2*l),axis=1)
        return isoverlap

    def _getrectangle(self,shape,cent,l):
        xmin = cent[0]-l
        xmax = cent[0]+l
        ymin = cent[1]-l
        ymax = cent[1]+l
        if xmin<0:
            xmin = 0
        if ymin<0:
            ymin = 0
        if xmax>shape[1]:
            xmax = shape[1]
        if ymax>shape[0]:
            ymax = shape[0]
        return [xmin, xmax, ymin, ymax]
    
    def _slicedmap(self,dmap,rec,cent,l):
        inrec = np.array(rec) - np.array([cent[0]-l,cent[0]-l,cent[1]-l,cent[1]-l])
        dmapslice = dmap[inrec[2]:inrec[3],inrec[0]:inrec[1]]
        return dmapslice
    
    def _mergerectangel(self,rec1,rec2):
        xmin = min(rec1[0],rec2[0])
        ymin = min(rec1[2],rec2[2])
        xmax = max(rec1[1],rec2[1])
        ymax = max(rec1[3],rec2[3])
        merged = [xmin,xmax,ymin,ymax]
        internal1 = self._mergerecsub(merged,rec1)
        internal2 = self._mergerecsub(merged,rec2)
        return merged, internal1, internal2
    
    def _mergerecsub(self,merged,rec):
        mxmin = merged[0]
        mymin = merged[2]
        internal = np.array(rec) - np.array([mxmin, mxmin, mymin, mymin])
        return internal
    
    def _crop(self,frame,r):
        return frame[r[2]:r[3],r[0]:r[1],:]
    
    def _tip(self,frame,rec,cent,dmap):
        if frame is None:
            return
        if frame.shape is not dmap.shape:
            dmap = self._slicedmap(dmap,rec,cent,self.c.sticklen)
        if not np.any(frame):
            return (rec[0]+rec[1])/2, (rec[2]+rec[3])/2
        distance = np.where(frame,dmap,0)
        tip_y, tip_x = np.unravel_index(np.argmax(distance),distance.shape)
        tip_y = tip_y.astype(int) + rec[2]
        tip_x = tip_x.astype(int) + rec[0]
        return tip_x, tip_y

    def _distancemap(self,r,c):
        rc, cc = [int(r/2),int(c/2)]
        ri,ci = np.ogrid[-rc:r-rc, -cc:c-cc]
        map = ri*ri + ci*ci
        return map






import scipy.signal

@dataclass
class CalcAccCnfg():
    savgol_window:int = 31
    savgol_poly:int = 6
    savgol_mode:str = 'nearest'
    savgol_pre_window:int = 11
    savgol_pre_poly:int = 3
    savgol_pre_mode:str = 'nearest'
    savgol_pre_sigma:int = 2

@dataclass
class CalcAccCnfgStick():
    savgol_window:int = 25
    savgol_poly:int = 4
    savgol_mode:str = 'nearest'
    savgol_pre_window:int = 15
    savgol_pre_poly:int = 3
    savgol_pre_mode:str = 'nearest'
    savgol_pre_sigma:int = 3

class CalcAcc():
    def __init__(self,pos):
        self.pos = pos
        self.frames = np.arange(self.pos.shape[0])
        self.dead = np.all(self.pos==0,axis=1)
        self.c = CalcAccCnfg()
    def setcnfg(self,c:CalcAccCnfg):
        self.c=c
    def getcnfg(self):
        return self.c
    def get_fullframe(self):
        def fill(alive):
            out = np.zeros(self.frames.shape[0])
            out[np.isin(self.frames,self.f)] = alive
            return out
        x = fill(self.xn)
        y = fill(self.yn)
        vx = fill(self.vx)
        vy = fill(self.vy)
        ax = fill(self.ax)
        ay = fill(self.ay)
        out = np.stack((x,y,vx,vy,ax,ay),axis=1)
        return out
    def get_only_alive(self,ix):
        if ix==0:
            return self.f, self.prex, self.xn, self.vx, self.ax
        if ix==1:
            return self.f, self.prey, self.yn, self.vy, self.ay
    def get_prefilter(self,ix):
        if ix==0:
            return (self.f, self.x, self.smoothx, 
            self.resx, self.borderx, self.outx)
        if ix==1:
            return (self.f, self.y, self.smoothy,
            self.resy, self.bordery, self.outy)
    def calc(self):
        self.f = self.frames[~self.dead]
        self.x = self.pos[:,0][~self.dead]
        self.y = self.pos[:,1][~self.dead]

        self.prex,self.resx,self.borderx,self.outx,self.smoothx = \
        self._rob_savgol(self.x)
        self.prey,self.resy,self.bordery,self.outy,self.smoothy = \
        self._rob_savgol(self.y)

        self.xn,self.yn,self.vx,self.vy,self.ax,self.ay\
        = self._interp_savgol(self.prex,self.prey)

    def _interp_savgol(self,x,y):
        wl = self.c.savgol_window
        po = self.c.savgol_poly
        md = self.c.savgol_mode
        xn = scipy.signal.savgol_filter(x,window_length=wl,
        polyorder=po,deriv=0,mode=md)
        vx = scipy.signal.savgol_filter(x,window_length=wl,
        polyorder=po,deriv=1,mode=md)
        ax = scipy.signal.savgol_filter(x,window_length=wl,
        polyorder=po,deriv=2,mode=md)

        yn = scipy.signal.savgol_filter(y,window_length=wl,
        polyorder=po,deriv=0,mode=md)
        vy = scipy.signal.savgol_filter(y,window_length=wl,
        polyorder=po,deriv=1,mode=md)
        ay = scipy.signal.savgol_filter(y,window_length=wl,
        polyorder=po,deriv=2,mode=md)

        return xn,yn,vx,vy,ax,ay

    def _rob_savgol(self,x):
        wl = self.c.savgol_pre_window
        po = self.c.savgol_pre_poly
        md = self.c.savgol_pre_mode
        sigma = self.c.savgol_pre_sigma

        new = copy.deepcopy(x)
        smooth = scipy.signal.savgol_filter(x,window_length=wl,
        polyorder=po,deriv=0,mode=md)
        res = smooth-x
        sd = res.std()
        border = sd*sigma
        out = np.abs(res)>border
        new[out]=smooth[out]
        return new,res,border,out,smooth


from guitools import FitPlotBase

class FitPlotOneAxis(FitPlotBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initPlot()
    def initPlot(self):
        v = np.array([[0,1],[0,1]])
        self.sty1 = dict(symbol=None,pen={'color':'c','width':2})
        self.sty2 = dict(pen=None,symbol='o',symbolSize=2,
            symbolPen={'color':'m','width':2})
        self.sty3 = dict(pen={'color':'w','width':1})
        self.sty4 = dict(pen=None,symbol='o',symbolSize=2,
            symbolPen={'color':'c','width':2})
        self.sty5 = dict(pen=None,symbol='o',symbolSize=2,
            symbolPen={'color':'y','width':2})

        self.p0 = self.pli[0].plot(v,**self.sty1)
        self.p1 = self.pli[1].plot(v,**self.sty1)
        self.p2 = self.pli[2].plot(v,**self.sty1)
        self.p2dot = self.pli[2].plot(v,**self.sty2)
        self.p3dot = self.pli[3].plot(v,**self.sty2)
        self.p4 = self.pli[4].plot(v,**self.sty1)
        self.p4dot0 = self.pli[4].plot(v,**self.sty2)
        self.p4dot1 = self.pli[4].plot(v,**self.sty5)
        self.p5dot0 = self.pli[5].plot(v,**self.sty2)
        self.p5dot1 = self.pli[5].plot(v,**self.sty4)
        self.p5line0 = self.pli[5].addLine(y=0,**self.sty3)
        self.p5line1 = self.pli[5].addLine(y=0,**self.sty3)
        # a,v,xn,xn_res,prex,pre_res
    def draw_diff(self,f,x,xn,vx,ax):
        self.p0.setData(f,ax)
        self.p1.setData(f,vx)
        self.p2.setData(f,xn)
        self.p2dot.setData(f,x)
        self.p3dot.setData(f,xn-x)
    def draw_pre(self,f,x,pre,res,border,out):
        self.p4.setData(f,pre)
        self.p4dot0.setData(f[out],x[out])
        self.p4dot1.setData(f[~out],x[~out])
        self.p5dot0.setData(f[~out],res[~out])
        self.p5dot1.setData(f[out],res[out])
        self.p5line0.setValue(border)
        self.p5line1.setValue(-border)
        
from PyQt5.QtWidgets import QWidget
from guitools import ConfigEditor

class FitPlotWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        self.l1 = QHBoxLayout()
        self.l0.addLayout(self.l1,3)
        self.plt = [FitPlotOneAxis(self) for i in range(2)]
        self.l1.addWidget(self.plt[0])
        self.l1.addWidget(self.plt[1])

        self.l2 = QVBoxLayout()
        self.l0.addLayout(self.l2,1)
        self.configedit = ConfigEditor()
        self.l2.addWidget(self.configedit)
        self.fin_button = QPushButton('finish')
        self.l2.addWidget(self.fin_button)

    def draw_diff(self,ix,*args):
        self.plt[ix].draw_diff(*args)
    def draw_pre(self,ix,*args):
        self.plt[ix].draw_pre(*args)
    def set_config(self,d):
        self.configedit.setdict(d)

class FitPlotTabs(QTabWidget):
    Finished = pyqtSignal(str)
    Updated = pyqtSignal(str,dict)
    def __init__(self,dianum,parent=None):
        super().__init__(parent)
        self.dianum = dianum
        self.initUI()
        self.left.fin_button.clicked.connect(lambda: self.finished('l'))
        self.left.configedit.UpdatedConfig.connect(lambda x: self.updated('l',x))
        self.right.fin_button.clicked.connect(lambda: self.finished('r'))
        self.right.configedit.UpdatedConfig.connect(lambda x: self.updated('r',x))
        for i in range(self.dianum):
            self.dias[i].fin_button.clicked.connect(lambda: self.finished('d'+str(i)))
            self.dias[i].configedit.UpdatedConfig.connect(lambda x: self.updated('d'+str(i),x))
    def initUI(self):
        self.left = FitPlotWidget()
        self.right = FitPlotWidget()
        self.dias = [FitPlotWidget() for i in range(self.dianum)]
        self.addTab(self.left,'left stick')
        self.addTab(self.right,'right stick')
        for i,dia in enumerate(self.dias):
            self.addTab(dia,'d'+str(i))
    def finished(self,key):
        self.Finished.emit(key)
    def updated(self,key,d):
        self.Updated.emit(key,d)
    def set_config(self,key,d):
        if key =='l':
            a = self.left
        elif key =='r':
            a = self.right
        elif key[0] == 'd':
            i = int(key[1:])
            a = self.dias[i]
        a.set_config(d)
    def draw_diff(self,key,*args):
        if key=='l':
            self.left.draw_diff(*args)
        if key=='r':
            self.right.draw_diff(*args)
        if key[0]=='d':
            i = int(key[1:])
            self.dias[i].draw_diff(*args)
    def draw_pre(self,key,*args):
        if key=='l':
            self.left.draw_pre(*args)
        if key=='r':
            self.right.draw_pre(*args)
        if key[0]=='d':
            i = int(key[1:])
            self.dias[i].draw_pre(*args)

class FitPlotControl():
    def __init__(self,lpos,rpos,dposlist):
        self.lpos = lpos
        self.rpos = rpos
        self.dpos = dposlist
        self.dianum = len(dposlist)
        self.window = FitPlotTabs(self.dianum)
        self.l_calc = CalcAcc(lpos)
        self.l_calc.setcnfg(CalcAccCnfgStick())
        self.window.set_config('l',asdict(self.l_calc.getcnfg()))
        self.l_calc.calc()
        self.draw('l')
        self.r_calc = CalcAcc(rpos)
        self.r_calc.setcnfg(CalcAccCnfgStick())
        self.window.set_config('r',asdict(self.r_calc.getcnfg()))
        self.r_calc.calc()
        self.draw('r')
        self.d_calc=[None for i in range(self.dianum)]
        for i in range(self.dianum):
            self.d_calc[i] = CalcAcc(dposlist[i])
            self.d_calc[i].calc()
            self.window.set_config('d'+str(i),asdict(self.d_calc[i].getcnfg()))
            self.draw('d'+str(i))
        self.window.Updated.connect(self.update)

    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.Finished
    def get_df(self):
        basename = ['_savgol_x','_savgol_y','_vx','_vy','_ax','_ay']
        lname = ['l'+n for n in basename]
        l = self.l_calc.get_fullframe()
        ldf = pd.DataFrame(l,columns=lname)
        rname = ['r'+n for n in basename]
        r = self.r_calc.get_fullframe()
        rdf = pd.DataFrame(r,columns=rname)
        d=[]
        ddf=[]
        for i in range(self.dianum):
            dname = ['d'+str(i)+n for n in basename]
            d.append(self.d_calc[i].get_fullframe())
            ddf.append(pd.DataFrame(d[i],columns=dname))
        dflist = [ldf,rdf]+ddf
        df = pd.concat(dflist,axis=1)
        return df
    def update(self,key,d):
        if key =='l':
            a = self.l_calc
        elif key =='r':
            a = self.r_calc
        elif key[0] == 'd':
            i = int(key[1:])
            a = self.d_calc[i]
        a.setcnfg(CalcAccCnfg(**d))
        a.calc()
        self.draw(key)
    def draw(self,key):
        if key=='l':
            a = self.l_calc
        elif key =='r':
            a = self.r_calc
        elif key[0] == 'd':
            i = int(key[1:])
            a = self.d_calc[i]
        for ix in range(2):
            self.window.draw_diff(key,ix,*a.get_only_alive(ix))
            self.window.draw_pre(key,ix,*a.get_prefilter(ix))


class TestFit():
    '''test class'''
    def __init__(self):
        self.df1 = pd.read_csv('./test7/testing_3.csv')
        self.df2 = pd.read_csv('./test7/testing_2_circle.csv')
        basename = ['tip_x','tip_y']
        lname = ['l'+n for n in basename]
        lpos = self.df1[lname].values
        rname = ['r'+n for n in basename]
        rpos = self.df1[rname].values
        dbasename = ['c_x','c_y']
        dposlist = []
        for i in range(2):
            dname = ['d'+str(i)+n for n in dbasename]
            dposlist.append(self.df2[dname].values)
        
        self.control = FitPlotControl(lpos,rpos,dposlist)
        self.window = self.control.get_window()
        self.window.show()
        self.control.finish_signal().connect(self.finish)
    def finish(self):
        print('a')
        res = self.control.get_df()
        df = pd.concat((self.df2,self.df1,res),axis=1)
        df.to_csv('./test7/testing_4.csv')



from guitools import GravityFitBase

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
        self.flpath = './test7/td_fall.mov'
        self.flld = Loader(self.flpath)
        self.dfpath = './test7/fall_circle.csv'
        self.df = pd.read_csv(self.dfpath,index_col=0)
        self.fit = GravityFitControl(self.flld,self.df[['dc_x','dc_y']].values)
        self.fit.get_window().show()
        self.fit.finish_signal().connect(self.a)
    def a(self):
        print(self.fit.get_g(),self.fit.get_frames())







class WrapState():
    '''
    'n' = 0 = neutral base
    'm' = 1 = recapture base
    'R' = 2 = right wrap
    'L' = 3 = left wrap
    'r' = 4 = right unwrap
    'l' = 5 = left unwrap
    '''
    d = {'n':0,'m':1,'R':2,
        'L':3,'r':4,'l':5}
    rev_d = {i:l for l,i in d.items()}

    def __init__(self):
        self.state = [0]
    def str2int(self,s):
        out = []
        for letter in s:
            out.append(WrapState.d[letter])
        return out
    def int2str(self,i):
        out = ''
        for n in i:
            out = out+WrapState.rev_d[n]
        return out
    def set_state(self,str_state):
        self.state = self.str2int(str_state)
    def get_state(self):
        return self.int2str(self.state)
    def R(self):
        last = self.state[-1]
        if last==1:
            self.state=[0]
        elif last==4:
            self.state=self.state[:-1]
        else:
            self.state.append(2)
    def L(self):
        last = self.state[-1]
        if last==1:
            self.state=[0]
        elif last==5:
            self.state=self.state[:-1]
        else:
            self.state.append(3)
    def r(self):
        last = self.state[-1]
        if last==0:
            self.state=[1]
        elif last==2:
            self.state=self.state[:-1]
        else:
            self.state.append(4)
    def l(self):
        last = self.state[-1]
        if last==0:
            self.state=[1]
        elif last==3:
            self.state=self.state[:-1]
        else:
            self.state.append(5)
    def swap(self):
        newstate = []
        for n in self.state:
            if n==0:
                newstate.append(1)
            elif n==1:
                newstate.append(0)
            elif n==2:
                newstate.append(5)
            elif n==3:
                newstate.append(4)
            elif n==4:
                newstate.append(3)
            elif n==5:
                newstate.append(2)
        self.state = newstate
    def passthrough(self):
        if self.state == [0]:
            self.state = [1]
        elif self.state == [1]:
            self.state = [0]
    def wrapnumber(self):
        num = 0
        for l in self.state:
            if l==0:
                num = 0
            elif l==1:
                num = -1
            elif l==2 or l==3:
                num = num+1
            elif l==4 or l==5:
                num = num-1
        return num


class WrapStateStore():
    operation_keys = ['R','L','r','l','s','p']
    def __init__(self,firstframe,lastframe):
        self.w = WrapState()
        self.initialstate = self.w.get_state()
        self.firstframe = firstframe
        self.lastframe = lastframe
        self.diff_states = {}
        self.backward = False

        self.ope = {'R':self.w.R,
            'L':self.w.L,
            'r':self.w.r,
            'l':self.w.l,
            's':self.w.swap,
            'p':self.w.passthrough}
    def get_states(self):
        states = self._makestates()
        return states
    def get_diffstates(self):
        return self.diff_states
    def get_state(self,fpos):
        self._goto(fpos)
        return self.w.get_state()
    def get_initialstate(self):
        return self.initialstate
    def get_wrapnum(self):
        states = self._makestates()
        temp = WrapState()
        wrapnum = np.zeros(self.lastframe-self.firstframe)
        for i,s in enumerate(states):
            temp.set_state(s)
            wrapnum[i] = temp.wrapnumber()
        return wrapnum
    def search(self,opekey):
        frames = [key for key,val in self.diff_states.items() if val == opekey]
        return sorted(frames)
    def set_state(self,str_state):
        self.initialstate=str_state
    def set_firstframe(self,first):
        self.firstframe=first
        self.diff_states = {key:item for key,item in self.diff_states.items()
            if key>=self.firstframe}
    def set_lastframe(self,last):
        self.lastframe = last
        self.diff_states = {key:item for key,item in self.diff_states.items()
            if key<self.lastframe}
    def R(self,fp):
        self.diff_states.update({fp:'R'})
    def r(self,fp):
        self.diff_states.update({fp:'r'})
    def L(self,fp):
        self.diff_states.update({fp:'L'})
    def l(self,fp):
        self.diff_states.update({fp:'l'})
    def swap(self,fp):
        self.diff_states.update({fp:'s'})
    def passthrough(self,fp):
        self.diff_states.update({fp:'p'})
    def clear(self,fp):
        self.diff_states.pop(fp,None)
    def clearafter(self,fp):
        l = [f for f in self.diff_states.keys() if f>fp]
        for f in l:
            self.clear(f)

    def _makestates(self):
        states = [self.initialstate for i 
            in range(self.firstframe,self.lastframe)]
        self.w.set_state(self.initialstate)
        for fp,opekey in sorted(self.diff_states.items()):
            self.ope[opekey]()
            states[fp-self.firstframe:] = [self.w.get_state() for i
                in range(fp,self.lastframe)]
        return states
    def _goto(self,fpos):
        self.w.set_state(self.initialstate)
        for fp,opekey in sorted(self.diff_states.items()):
            if fp>fpos:
                return
            self.ope[opekey]()

class ObjectOnString():
    def __init__(self,key:str,pos):
        '''pos: ndarray [framenum,2] (x,y)'''
        self.key = key
        self.pos = pos
        self.frame = np.arange(pos.shape[0])
        self.dead = np.all(pos==0,axis=1)
        self.first = np.nonzero(~self.dead)[0].min()
        self.last = np.nonzero(~self.dead)[0].max()


class Angle():
    def __init__(self,l:ObjectOnString,r:ObjectOnString,c:ObjectOnString,
            first:int,last:int,initialwrap:str):
        self.l = l
        self.r = r
        self.c = c
        self.first = first
        self.last = last
        self.wrap = WrapStateStore(self.first,self.last)
        self.wrap.set_state(initialwrap)
        self.dead = np.logical_or(self.l.dead,self.r.dead,self.c.dead)
        self.calc_angle()
    def get_keys(self):
        return [o.key for o in [self.l, self.r, self.c]]
    def get_objects(self):
        return [self.l,self.r,self.c]
    def get_wrap_diff(self):
        return self.wrap.get_diffstates()
    def get_wrapstates(self):
        '''this is first:last cropped'''
        return self.wrap.get_states()
    def get_wrapstate(self,fpos):
        return self.wrap.get_state(fpos)
    def get_phi_theta(self):
        return self.phi,self.theta
    def set_initialwrap(self,wrap:str):
        self.wrap.set_state(wrap)
        self.theta_correction()
    def auto_initialwrap(self,key):
        self.set_initialwrap('n')
        if key=='fly':
            if self.theta[self.first]>np.pi:
                wrap = 'm'
            else:
                wrap = 'n'
        elif key=='land':
            if self.theta[self.first]>np.pi:
                wrap = 'n'
            else:
                wrap = 'm'
        self.set_initialwrap(wrap)
    def autofill_wrap(self,fpos=None):
        if fpos is None:
            fpos = self.first
        self.wrap.clearafter(fpos)
        iswrap,isunwrap = self.is_crossing()
        iswrap[:fpos+1]=False
        isunwrap[:fpos+1]=False
        if np.any(iswrap):
            wf = np.nonzero(iswrap)[0]
            for i in wf:
                self.wrap.R(i)
        if np.any(isunwrap):
            uf = np.nonzero(isunwrap)[0]
            for i in uf:
                self.wrap.r(i)
        self.theta_correction()
    def edit_wrap(self,fpos:int,opekey:str):
        ope = {'R':self.wrap.R,
            'L':self.wrap.L,
            'r':self.wrap.r,
            'l':self.wrap.l,
            's':self.wrap.swap,
            'p':self.wrap.passthrough,
            'c':self.wrap.clear}
        ope[opekey](fpos)
        self.theta_correction()
    def set_firstframe(self,first:int):
        self.first = first
        self.wrap.set_firstframe(first)
        self.theta_correction()
    def set_lastframe(self,last:int):
        old = self.last
        self.last = last
        self.wrap.set_lastframe(last)
        if old<self.last:
            self.autofill_wrap(old)
        self.theta_correction()
    def theta_correction(self):
        correction = np.floor(self.rawtheta[self.first]/(2*np.pi))
        temptheta = self.rawtheta - 2*np.pi*correction
        if len(self.wrap.get_wrapnum())==0:
            a=0
        iniwrapnum = self.wrap.get_wrapnum()[0]
        self.theta = temptheta + 2*np.pi*iniwrapnum
        passfr = self.wrap.search('p')
        for fr in passfr:
            if self.theta[fr]>0:
                self.theta[fr:] -= 2*np.pi
            elif self.theta[fr]<0:
                self.theta[fr:] += 2*np.pi
        swapfr = self.wrap.search('s')
        for fr in swapfr:
            self.theta[fr:] = -self.theta[fr:]

    def calc_angle(self):
        # everything is [l,r]
        self.phi_raw = [0,0] # 0-2pi.
        self.dphi   = [0,0]
        self.up     = [0,0]
        self.down   = [0,0]
        self.phi    = [0,0] # continuous radian

        # y -1* due to decartesian v.s. opencv coords
        for i in range(2):
            if i==0:
                hx = self.l.pos[:,0]
                hy = -self.l.pos[:,1]
            else:
                hx = self.r.pos[:,0]
                hy = -self.r.pos[:,1]
            dx = self.c.pos[:,0]
            dy = -self.c.pos[:,1]
            self._calc_angle_sub(hx,hy,dx,dy,i)
        self.rawtheta = self.phi[0] - self.phi[1]
        self.theta_correction()
    def _calc_angle_sub(self,hx,hy,dx,dy,i):
        vec = np.stack((hx-dx, hy-dy),axis=1)
        arctan = np.where(vec[:,0]==0,0,np.arctan(vec[:,1]/vec[:,0]))
        phi_raw = np.where(vec[:,0]>=0,arctan,arctan+np.pi)
        self.phi_raw[i] = np.where(phi_raw>=0,phi_raw,phi_raw+2*np.pi)

        self.dphi[i] = np.zeros_like(self.phi_raw[i])
        self.dphi[i][1:] = self.phi_raw[i][1:]-self.phi_raw[i][:-1]

        up = np.nonzero(self.dphi[i]>np.pi)[0]
        self.up[i] = up[np.isin(up-1,self.dead,invert=True)]
        down = np.nonzero(self.dphi[i]<-np.pi)[0]
        self.down[i] = down[np.isin(down-1,self.dead,invert=True)]

        rot = np.zeros_like(self.phi_raw[i],dtype=int)
        for j in self.up[i]:
            rot[j:] -= 1
        for j in self.down[i]:
            rot[j:] += 1

        self.phi[i] = self.phi_raw[i] + 2*np.pi*rot

    def is_flying(self):
        logic1 = np.logical_and(-1*np.pi<self.theta,self.theta<=0)
        logic2 = np.logical_and(np.pi>=self.theta,self.theta>0)
        logic1 = logic1[self.first:self.last]
        logic2 = logic2[self.first:self.last]
        wstates = self.wrap.get_states()
        logic3 = np.array(wstates)=='m'
        logic4 = np.array(wstates)=='n'
        fly1 = np.logical_and(logic1,logic3)
        fly2 = np.logical_and(logic2,logic4)
        fly_crop = np.logical_or(fly1,fly2)
        is_flying = np.zeros_like(self.theta,dtype=bool)
        is_flying[self.first:self.last]=fly_crop
        is_touching = np.zeros_like(self.theta,dtype=bool)
        is_touching[self.first:self.last]=~fly_crop
        return is_flying,is_touching
    def is_crossing(self):
        step = np.floor(self.theta/(np.pi * 2))
        dstep = step[1:]-step[:-1]
        dstep = np.concatenate((np.zeros(1,dtype=int),dstep))
        dstep[self.first]=0
        dstep[:self.first]=0
        dstep[self.last:]=0
        is_wrapping = dstep==1
        is_unwrapping = dstep==-1
        return is_wrapping,is_unwrapping
    def takeoff_frame(self):
        fly = self.is_flying()[0]
        if np.all(~fly):
            return None
        frame = np.nonzero(fly)[0].min()
        return frame
    def landing_frame(self):
        land = self.is_flying()[1]
        if np.all(~land):
            return None
        frame = np.nonzero(land)[0].min()
        return frame

class AngleSet():
    def __init__(self):
        self.angles = []
        self.flys = []
    def get(self):
        return self.angles,self.flys
    def by_object(self,pos:str,key:str):
        ak = [angle.get_keys() for angle in self.angles]
        fk = [angle.get_keys() for angle in self.flys]
        d = {'l':0,'r':1,'c':2}
        ix = d[pos]
        aix = [keys[ix]==key for keys in ak]
        fix = [keys[ix]==key for keys in fk]
        filt_a = [self.angles[i] for i in range(len(aix)) if aix[i]]
        filt_f = [self.flys[i] for i in range(len(fix)) if fix[i]]
        new = AngleSet()
        new.add_angles(*filt_a)
        new.add_flys(*filt_f)
        return new
    def by_frame(self,frame:int):
        first_a = [angle.first for angle in self.angles]
        last_a = [angle.last for angle in self.angles]
        first_f = [angle.first for angle in self.flys]
        last_f = [angle.last for angle in self.flys]

        ix_a_first = [frame>=f for f in first_a]
        ix_a_last = [frame<f for f in last_a]
        ix_a = [f and l for f,l in zip(ix_a_first,ix_a_last)]
        ix_f_first = [frame>=f for f in first_f]
        ix_f_last = [frame<f for f in last_f]
        ix_f = [f and l for f,l in zip(ix_f_first,ix_f_last)]

        filt_a = [self.angles[i] for i in range(len(ix_a)) if ix_a[i]]
        filt_f = [self.flys[i] for i in range(len(ix_f)) if ix_f[i]]
        new = AngleSet()
        new.add_angles(*filt_a)
        new.add_flys(*filt_f)
        return new
    def add_angles(self,*angles):
        for angle in angles:
            self.angles.append(angle)
    def add_flys(self,*flys):
        for angle in flys:
            self.flys.append(angle)
    def erase_after(self,fpos):
        first_a = [angle.first for angle in self.angles]
        first_f = [angle.first for angle in self.flys]
        aix = [f<=fpos for f in first_a]
        fix = [f<=fpos for f in first_f]
        self.angles = [self.angles[i] for i in range(len(aix)) if aix[i]]
        self.flys = [self.flys[i] for i in range(len(fix)) if fix[i]]
    def next_landing_takeoff(self):
        landing_frames = [a.landing_frame() for a in self.flys]
        takeoff_frames = [a.takeoff_frame() for a in self.angles]
        nolanding = [i is None for i in landing_frames]
        notakeoff = [i is None for i in takeoff_frames]
        if all(nolanding) and all(notakeoff):
            return None,None,None
        if not all(nolanding):
            firstlanding = min(np.array(landing_frames)[~np.array(nolanding)])
            first = firstlanding
        if not all(notakeoff):
            firsttakeoff= min(np.array(takeoff_frames)[~np.array(notakeoff)])
            first = firsttakeoff
        if (not all(nolanding)) and (not all(notakeoff)):
            first = min([firstlanding,firsttakeoff])

        l_ix = np.array(landing_frames)==first
        t_ix = np.array(takeoff_frames)==first
        landings = list(np.array(self.flys)[l_ix])
        takeoffs = list(np.array(self.angles)[t_ix])
        # landings/takeoffs is [] if none
        return landings,takeoffs,first
    def get_wraps(self,fly=True):
        '''only works after by_object'''
        wraps = {}
        for angle in self.angles:
            wraps.update(angle.get_wrap_diff())
        if fly:
            for angle in self.flys:
                takeoff = angle.first
                landing = angle.last
                wraps.update({takeoff:'takeoff',landing:'landing'})
                passthrough = angle.wrap.search('p')
                for p in passthrough:
                    wraps.update({p:'p'})
        return wraps
    def get_wrapstates(self,framenum):
        '''only works after by_object'''
        wraps = ['' for i in range(framenum)]
        for angle in self.angles:
            wraps[angle.first:angle.last]=angle.get_wrapstates()
        return wraps
    def get_wrapstate(self,fpos):
        '''onlyworks after by_object and by_frame'''
        if len(self.angles)==0:
            return None
        w = self.angles[0].get_wrapstate(fpos)
        return w




class ObjectChain():
    def __init__(self,lpos,rpos,dposlist):
        self.objects = []
        self.objects.append(ObjectOnString('l',lpos))
        for i in range(len(dposlist)):
            self.objects.append(ObjectOnString('d'+str(i),dposlist[i]))
        self.objects.append(ObjectOnString('r',rpos))
        
        self.framenum = lpos.shape[0]
        # [[frame,...],[[chain],...],[[flying],...],[[absent],...]]
        self.diff_states = [[],[],[],[]]

        firsts = [o.first for o in self.objects]
        lasts = [o.last for o in self.objects]
        self.allfirst = max(firsts)
        self.alllast = min(lasts)

        self.initialize_chain()

    def initialize_chain(self):
        keys = [o.key for o in self.objects]
        self.add_diff_state([0,[],[],keys])
        self.add_diff_state([self.allfirst,keys,[],[]])
        self.add_diff_state([self.alllast,[],[],keys])
    def get_objects(self,*keys):
        obj_keys = [obj.key for obj in self.objects]
        out = []
        for key in keys:
            out.append(self.objects[obj_keys.index(key)])
        return out
    def get_keys(self):
        obj_keys = [obj.key for obj in self.objects]
        return obj_keys
    def get_state(self,frame):
        bef = [f for f in self.diff_states[0] if f<=frame]
        ix = np.argmax(bef)
        current = copy.deepcopy([self.diff_states[i][ix] for i in range(4)])
        current[0]=frame
        for i in range(3):
            current[i+1] = list(current[i+1])
        return current
    def add_diff_state(self,state):
        frame = state[0]
        if frame in self.diff_states[0]:
            ix = self.diff_states[0].index(frame)
            self.diff_states[1][ix]=list(state[1])
            self.diff_states[2][ix]=list(state[2])
            self.diff_states[3][ix]=list(state[3])
            return
        self.diff_states[0].append(state[0])
        self.diff_states[1].append(list(state[1]))
        self.diff_states[2].append(list(state[2]))
        self.diff_states[3].append(list(state[3]))
        self._sort_diff()
    def erase_after(self,fpos):
        self._sort_diff()
        for i, fr in enumerate(self.diff_states[0]):
            if fr>fpos:
                ix = i
                break
            ix = None
        if ix is None:
            return
        for i in range(4):
            self.diff_states[i] = self.diff_states[i][:ix]
        
    def get_states(self):
        self._sort_diff()
        frame,chain,fly,absent = self.diff_states
        chains = [[] for i in range(self.framenum)]
        flys = [[] for i in range(self.framenum)]
        for i,(fs,fe) in enumerate(zip(frame[:-1],frame[1:])):
            chains[fs:fe]=[chain[i] for j in range(fs,fe)]
            flys[fs:fe]=[fly[i] for j in range(fs,fe)]
        chains[frame[-1]:]=[chain[-1] for j in range(frame[-1],self.framenum)]
        flys[frame[-1]:]=[fly[-1] for j in range(frame[-1],self.framenum)]
        return chains,flys
    def get_diff_states(self):
        self._sort_diff()
        return self.diff_states
        
    def _sort_diff(self):
        cp = copy.deepcopy(self.diff_states)
        frame = cp[0]
        ix = np.argsort(frame)
        for i in range(4):
            s = [cp[i][j] for j in ix]
            cp[i] = s
        self.diff_states = cp
    def takeoff(self,frame,key):
        # repeatable
        current = self.get_state(frame)
        current[1].remove(key)
        current[2].append(key)
        self.add_diff_state(current)
    def landing(self,frame,lkey,rkey,ckey):
        # repeatable
        current = self.get_state(frame)
        current[2].remove(ckey)
        chain = current[1]
        insert_ix = chain.index(rkey)
        chain.insert(insert_ix,ckey)
        self.add_diff_state(current)

class AngleAssigner():
    def __init__(self,lpos,rpos,dposlist:list,stickpos:StickPosition):
        self.oc = ObjectChain(lpos,rpos,dposlist)
        self.dianum = len(dposlist)
        self.angles = AngleSet()
        self.framenum = lpos.shape[0]
        self.stickpos = stickpos
    
    def get_results(self):
        # # wrap_initial (frame,wraplist(d0,d1,...))
        # ini_wrap_frame = self.initial_wraps_frame
        # ini_keys_wrap_dict = self.initial_wraps_keys
        # ini_wrap_list = [ini_keys_wrap_dict['d'+str(i)] for i in range(self.dianum)]
        # # wrap_diff (wraplist (d0,d1,...))
        # wrap_diff_keys_list,wrap_diff = self.get_wrap_diff()
        # wrap_diff_list = []
        # for wrapdict in wrap_diff:
        #     out = [[],[]] #framelist,opekeylist
        #     out[0]=list(sorted(wrapdict.keys()))
        #     out[1] = [wrapdict[f] for f in out[0]]
        #     wrap_diff_list.append(out)
        # wrap states
        wrap_state_list = self.get_wrapstates_list()
        name = []
        arr = np.zeros((self.framenum,self.dianum),dtype=object)
        for i,st in enumerate(wrap_state_list):
            name.append('d'+str(i)+'_wrapstate')
            arr[:,i] = st
        wrap_df = pd.DataFrame(arr,columns=name)
        # chain_diff
        chain_diff = self.oc.get_diff_states()
        # theta,phi
        thetalist,philist = self.get_theta_phi()
        basename = ['_theta','_phi0','_phi1']
        dflist = []
        for i,(theta,phi) in enumerate(zip(thetalist,philist)):
            name = ['d'+str(i)+n for n in basename]
            arr = np.concatenate((np.expand_dims(theta,1),phi),axis=1)
            dflist.append(pd.DataFrame(arr,columns=name))
        theta_phi_df = pd.concat(dflist,axis=1)

        return wrap_df,chain_diff,theta_phi_df

    def get_wrap_diff(self):
        keylist = []
        wrapslist= []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wraps(fly=True)
            keylist.append(key)
            wrapslist.append(wraps)
        return keylist,wrapslist
    def get_theta(self):
        thetadict = {}
        for i in range(self.dianum):
            key  = 'd'+str(i)
            theta = np.zeros(self.framenum)
            angles,_ = self.angles.by_object('c',key).get()
            for angle in angles:
                theta[angle.first:angle.last]=angle.theta[angle.first:angle.last]
            thetadict[key]=theta
        return thetadict
    def get_theta_phi(self):
        thetalist = []
        philist = []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            theta = np.zeros(self.framenum)
            phi = np.zeros((self.framenum,2))
            angles,_ = self.angles.by_object('c',key).get()
            for angle in angles:
                theta[angle.first:angle.last]=angle.theta[angle.first:angle.last]
                phi[angle.first:angle.last,0]=angle.phi[0][angle.first:angle.last]
                phi[angle.first:angle.last,1]=angle.phi[1][angle.first:angle.last]
            thetalist.append(theta)
            philist.append(phi)
        return thetalist,philist

    def get_wrapstates_list(self):
        wraps_list= []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wrapstates(self.framenum)
            wraps_list.append(wraps)
        return wraps_list
    def get_wrapstates(self):
        wraps_dict= {}
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wrapstates(self.framenum)
            wraps_dict[key]=wraps
        return wraps_dict
    def initialize(self,chain,wraps):
        '''example.
        chain: ['l','d1','d0','r']
        wraps: ['n','nR'] (d1=n,d0=nR)
        return False if invalid input
        '''
        # for readout
        self.initial_wraps_keys = {key:wrap for key,wrap in zip(chain[1:-1],wraps)}
        self.initial_wraps_frame = self.oc.allfirst

        if wraps==['']:
            wraps = []
        if len(chain)!=len(set(chain)):
            return False
        for c in chain:
            if c not in self.oc.get_keys():
                return False
        if chain[0]!='l':
            return False
        if chain[-1]!='r':
            return False
        if len(chain)-2 != len(wraps):
            return False

        fly = self.oc.get_keys()
        for key in chain:
            fly.remove(key)
        state = [self.oc.allfirst,chain,fly,[]]
        self.oc.add_diff_state(state)
        self.oc_to_angles(self.oc.allfirst)
        for i,c in enumerate(chain[1:-1]):
            wrap = wraps[i]
            angle,_ = self.angles.by_object('c',c).by_frame(self.oc.allfirst).get()
            angle = angle[0]
            angle.set_initialwrap(wrap)
        return True
    
    def forward_repeat(self,fpos=None):
        if fpos is None:
            fpos = self.oc.allfirst
        framepos = fpos

        self.angles.erase_after(framepos)
        self.oc.erase_after(framepos)
        angles,flys = self.angles.by_frame(framepos).get()
        for a in angles:
            a.set_lastframe(self.framenum)
        for a in flys:
            a.set_lastframe(self.framenum)

        while framepos is not False:
            next = self.scan_forward(framepos)
            framepos = next
    
    def wrap_and_forward(self,frame,diaix,opekey):
        if frame not in range(self.oc.allfirst,self.oc.alllast):
            return False
        if 'd'+str(diaix) not in self.oc.get_keys():
            return False
        self.edit_wrap(frame,diaix,opekey)
        if opekey in ['R','L','r','l','c']:
            self.forward_repeat(frame)
        elif opekey in ['p']:
            self.forward_repeat(frame-1)

        return True

    def scan_forward(self,fpos):
        landings,takeoffs,frame = self.angles.by_frame(fpos).next_landing_takeoff()
        if frame is None:
            return False
        for land in landings:
            self.oc.landing(frame,*land.get_keys())
        for tkof in takeoffs:
            self.oc.takeoff(frame,tkof.get_keys()[2])
        self.oc_to_angles(frame)
        return frame

    def oc_to_angles(self,fpos):
        _,chain,fly,_ = self.oc.get_state(fpos)
        newkeysets = []
        if len(chain)>2:
            newkeysets = list(zip(chain[:-2],chain[2:],chain[1:-1]))
        newflysets = []
        if len(chain)>1:
            for flykey in fly:
                newflysets += list(zip(chain[:-1],chain[1:],[flykey for i in range(len(chain)-1)]))
        
        currentangles,currentflys = self.angles.by_frame(fpos).get()
        currentkeys = [angle.get_keys() for angle in currentangles]
        currentflykeys = [angle.get_keys() for angle in currentflys]

        for angle in currentangles:
            if angle.get_keys() not in newkeysets:
                angle.set_lastframe(fpos)
        for angle in currentflys:
            if angle.get_keys() not in newflysets:
                angle.set_lastframe(fpos)
        
        for keys in newkeysets:
            if keys not in currentkeys:
                l,r,c = self.oc.get_objects(*keys)
                newangle = Angle(l,r,c,fpos,self.oc.alllast,'n')
                previouswrap = self.angles.by_frame(fpos-1).by_object('c',keys[2]).get_wrapstate(fpos-1)
                if previouswrap is None:
                    newangle.auto_initialwrap('land')
                else:
                    newangle.set_initialwrap(previouswrap)
                newangle.autofill_wrap(fpos)
                self.angles.add_angles(newangle)
        for keys in newflysets:
            if keys not in currentflykeys:
                l,r,c = self.oc.get_objects(*keys)
                newangle = Angle(l,r,c,fpos,self.oc.alllast,'n')
                newangle.auto_initialwrap('fly')
                newangle.autofill_wrap(fpos)
                self.angles.add_flys(newangle)
        self.set_swap()

    def set_swap(self):
        swapframes = self.stickpos.swap_frame()
        for frame in swapframes:
            angles = self.angles.by_frame(frame).get()[0]
            for angle in angles:
                angle.edit_wrap(frame,'s')

    def edit_wrap(self,frame,diaix,opekey):
        angle,_ = self.angles.by_frame(frame).by_object('c','d'+str(diaix)).get()
        angle = angle[0]
        if opekey in ['R','L','r','l','c']:
            angle.set_lastframe(self.framenum)
            angle.edit_wrap(frame,opekey)
        elif opekey in ['p']:
            temp = self.angles.by_frame(frame-1).by_object('c','d'+str(diaix))
            _,fly = temp.by_object('l',angle.get_keys()[0]).get()
            fly = fly[0]
            fly.set_lastframe(self.framenum)
            fly.edit_wrap(frame,opekey)





from guitools import ChainAssignWidgetBase
from visualize import Drawing

class ChainAssignWidget(ChainAssignWidgetBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.drawing = Drawing(self.pli)
    def show_wraps(self,keylist,wraplist):
        txt = ''
        for i in range(len(keylist)):
            key = keylist[i]
            txt += f'{key}:\n'
            d = wraplist[i]
            sortedkey = sorted(d.keys())
            for k in sortedkey:
                txt += f'   {k}: {d[k]}\n'
        self.lb1.setText(txt)
    def show_initialframe(self,frame):
        self.sub_i.lb0.setText(f'initial frame: {frame}')
    def update_subwindow_frame(self,frame):
        self.sub_w.tb1.setText(str(frame))
    def update_subwindow_diaix(self,diaix):
        self.sub_w.tb2.setText(str(diaix))


from PyQt5.QtCore import Qt

class ChainAssignControl(ViewControlBase):
    def __init__(self,loader:Loader,lpos,
            rpos,dposlist,stickpos:StickPosition):
        super().__init__()
        self.ld = loader
        self.calc = AngleAssigner(lpos,rpos,dposlist,stickpos)
        self.window = ChainAssignWidget()

        self.window.drawing.set_fpos(self.ld.framenum)
        keys = self.calc.oc.get_keys()
        self.window.drawing.set_objectkeys(keys)
        objects = self.calc.oc.get_objects(*keys)
        pos = {}
        for key,object in zip(keys,objects):
            pos[key] = object.pos
        self.window.drawing.set_positions(pos)

        self.wrapframes = []
        self.fpos = 0

        self.change_fpos(self.calc.oc.allfirst)
        self.window.show_initialframe(self.calc.oc.allfirst)

        self.window.KeyPressed.connect(self.keyinterp)
        self.window.sub_i.Entered.connect(self.initialize)

    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.fin_b.clicked
    def get_results(self):
        return self.calc.get_results()
    def initialize(self,chain,wrap):
        success = self.calc.initialize(chain,wrap)
        if not success:
            self.window.sub_i.lb_warn.setText('invalid input')
            return
        self.calc.forward_repeat()
        self.get_wraps()
        self.set_drawing()
        self.window.sub_w.Entered.connect(self.wrapedit)
        self.window.tab.setTabEnabled(1,True)
        self.window.tab.setTabEnabled(0,False)
        self.change_fpos(self.fpos)
    def wrapedit(self,frame,diaix,opekey):
        success = self.calc.wrap_and_forward(frame,diaix,opekey)
        if not success:
            self.window.sub_w.lb_warn.setText('invalid input')
            return
        self.get_wraps()
        self.set_drawing()
        self.change_fpos(self.fpos)
    def set_drawing(self):
        self.window.drawing.clear()
        self.window.drawing.set_fpos(self.ld.framenum)
        keys = self.calc.oc.get_keys()
        self.window.drawing.set_objectkeys(keys)
        objects = self.calc.oc.get_objects(*keys)
        pos = {}
        for key,object in zip(keys,objects):
            pos[key] = object.pos
        self.window.drawing.set_positions(pos)
        self.window.drawing.set_chain(*self.calc.oc.get_states())
        self.window.drawing.set_wrap(self.calc.get_wrapstates(),self.calc.get_theta())
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        self.window.update_subwindow_frame(self.fpos)
        self.window.setcvimage(self.ld.getframe(self.fpos))
        self.window.drawing.show_fpos(self.fpos)
        self.window.drawing.show_positions(self.fpos)
        self.window.drawing.show_string(self.fpos)
        self.window.drawing.show_wrap(self.fpos)
        self.window.blockSignals(False)
    def get_wraps(self):
        keylist,wraplist = self.calc.get_wrap_diff()
        self.wrapframes = []
        for wrapd in wraplist:
            self.wrapframes += list(wrapd.keys())
        self.wrapframes = list(sorted(set(self.wrapframes)))
        self.window.show_wraps(keylist,wraplist)
        self.wraplistofdict = wraplist
    def keyinterp(self, key):
        super().keyinterp(key)
        if key==Qt.Key_N:
            self.nextwrap()
        if key==Qt.Key_P:
            self.previouswrap()
    def nextwrap(self):
        temp = [f for f in self.wrapframes if f>self.fpos]
        if len(temp) == 0:
            return
        newf = min(temp)
        diaix = 0
        for i,d in enumerate(self.wraplistofdict):
            if newf in d.keys():
                diaix = i
        self.change_fpos(newf)
        self.window.update_subwindow_diaix(diaix)
    def previouswrap(self):
        temp = [f for f in self.wrapframes if f<self.fpos]
        if len(temp) == 0:
            return
        newf = max(temp)
        diaix = 0
        for i,d in enumerate(self.wraplistofdict):
            if newf in d.keys():
                diaix = i
        self.change_fpos(newf)
        self.window.update_subwindow_diaix(diaix)



class TestChain():
    '''test class'''
    def __init__(self):
        impath = './test7/td2.mov'
        self.ld = Loader(impath)
        dfpath = './test7/testing_4.csv'
        self.df = pd.read_csv(dfpath,index_col=0)
        basename = ['_savgol_x','_savgol_y']
        lname = ['l'+n for n in basename]
        lpos = self.df[lname].values
        rname = ['r'+n for n in basename]
        rpos = self.df[rname].values
        dposlist = []
        for i in range(2):
            dname = ['d'+str(i)+n for n in basename]
            dpos = self.df[dname].values
            dposlist.append(dpos)
        stickpos = StickPosition(lpos.shape[0])
        stickpos.loadchanges_array(np.array([[427,1,2]]))
        self.ca = ChainAssignControl(self.ld,lpos,rpos,dposlist,stickpos)
        self.ca.get_window().show()
        self.ca.finish_signal().connect(self.a)
    def a(self):
        (wrap_states_df,
        chain_diff,theta_phi_df) = self.ca.get_results()
        print(f'wrapstates:\n{wrap_states_df.iloc[600:605,:]}\nchain_diff:\n{chain_diff}\n \
        theta_phi_df:\n{theta_phi_df.iloc[600:605,:]}')
        txt = json.dumps({
        'chain_diff':chain_diff
        },cls=NumpyEncoder,indent=4)
        txtpath = './test7/testing_5.txt'
        dfpath = './test7/testing_5.csv'
        with open(txtpath,mode='w') as f:
            f.write(txt)
        df = self.df.reset_index()
        df = pd.concat((df,theta_phi_df),axis=1)
        df = pd.concat((df,wrap_states_df),axis=1)
        print(df.columns)
        df.to_csv(dfpath,index_label=False)

from json import JSONEncoder
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return JSONEncoder.default(self, obj)


# from matplotlib import pyplot as plt

class ForceCalc():
    def __init__(self,grav,acc,phi,theta):
        self.grav = grav
        self.acc = acc
        self.phi = phi
        self.theta = theta
        self.pre_inverse_y()
        self._calc_force()
        self._calc_torque()
        # self.test()
        self.post_inverse_y()

    def get_results(self):
        arr = np.stack((self.forcex,self.forcey,self.fnorm,self.fangle,
        self.tension_l,self.tension_r,
        self.torque,self.tl_e,self.tr_e,self.dT_e),axis=1)
        return arr

    def post_inverse_y(self):
        self.tension_l*=-1
        self.tension_r*=-1
        self.torque*=   -1
    def pre_inverse_y(self):
        # for cartesian
        self.phi = -self.phi
    def _calc_force(self):
        ''' calcualte force from string(acc - g). unit: mg'''
        self.forcex = 0
        self.forcey = 0
        ax, ay = self.acc[:,0],self.acc[:,1]
        gx, gy, gnorm = self.grav
        self.forcex = (ax - gx)/gnorm
        self.forcey = (ay - gy)/gnorm
        self.fnorm = np.linalg.norm(np.stack((self.forcex,self.forcey),axis=1),axis=1)

        fangle = np.arctan(self.forcey/self.forcex)
        fangle = np.where(self.forcex>=0,fangle,fangle+np.pi)
        self.fangle = np.where(fangle>=0,fangle,fangle+2*np.pi)
    def _calc_torque(self):
        ''' calculate torque from force and string angle.
        unit: mgR '''
        theta = np.squeeze(self.theta)
        theta1 = self.phi[:,0]-self.fangle
        theta2 = self.fangle-self.phi[:,1]

        dead = theta==0

        self.tension_l = self.fnorm * np.sin(theta2)/np.sin(theta)
        self.tension_r = self.fnorm * np.sin(theta1)/np.sin(theta)
        self.torque = self.tension_r - self.tension_l

        self.dT_e = np.abs(1/np.sin(theta/2))
        self.tl_e = np.abs(1/np.sin(theta))
        self.tr_e = np.abs(1/np.sin(theta))

        self.tension_l[dead]=0
        self.tension_r[dead]=0
        self.torque[dead]=0
        self.dT_e[dead]=0
        self.tl_e[dead]=0
        self.tr_e[dead]=0
    # def test(self):
    #     fig = plt.figure()
    #     ax = fig.subplots(2,1)
    #     ax[0].plot(self.phi[:,0]/(2*np.pi))
    #     ax[0].plot(self.phi[:,1]/(2*np.pi))
    #     ax[0].plot(self.fangle/(2*np.pi))
    #     ax[1].plot(self.forcex[:]/(2*np.pi))
    #     ax[1].plot(self.forcey[:]/(2*np.pi))
    #     plt.show()

class ForceCalcControl():
    def __init__(self,grav,acclist,philist,thetalist):
        self.grav = grav
        self.dianum = len(acclist)
        self.calc = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            self.calc[i]=ForceCalc(self.grav,acclist[i],philist[i],
                thetalist[i])
    def get_df(self):
        dflist = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            key = 'd'+str(i)
            basename = ['_force_x','_force_y','_fnorm','_fangle','_tension_l','_tension_r',
            '_torque','_tl_e','_tr_e','_dT_e']
            name = [key+n for n in basename]
            array = self.calc[i].get_results()
            dflist[i] = pd.DataFrame(array,columns=name)
        df = pd.concat(dflist,axis=1)
        return df

class TestForce():
    def __init__(self):
        dfpath = './test7/testing_5.csv'
        df = pd.read_csv(dfpath,index_col=0)
        self.dianum = 2
        grav = (-0.0034861023477654935, 0.24583765706667549, 0.24586237317168205)
        acc = [0 for i in range(self.dianum)]
        phi = [0 for i in range(self.dianum)]
        theta = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            key = 'd'+str(i)
            acc[i] = df[[key+'_ax',key+'_ay']].values
            phi[i] = df[[key+'_phi0',key+'_phi1']].values
            theta[i] = df[[key+'_theta']].values
        
        self.calc=ForceCalcControl(grav,acc,phi,theta)
        res = self.calc.get_df()
        dfnewpath = './test7/testing_6.csv'
        dfnew = pd.concat((df,res),axis=1)
        dfnew.to_csv(dfnewpath)

import scipy.optimize

class TensionOptimization():
    def __init__(self,chain,force,phi,tl,tr,tl_e,tr_e):
        self.string_vectors = [self._phi2vector(p[:,0],p[:,1]) for p in phi]
        self.initial_tension = self._initial_tension(chain,force,tl,tr,tl_e,tr_e)
        self.chain = chain
        self.force = force
        self.optimized,self.optimized_force,\
        self.new_tl,self.new_tr,self.new_torque = self._optimize()
    def get(self):
        return self.optimized, self.optimized_force,\
        self.new_tl,self.new_tr,self.new_torque
    def _optimize(self):
        out = copy.deepcopy(self.initial_tension)
        out_force = copy.deepcopy(self.force)
        dianum = len(self.force)
        new_torque = np.zeros((self.force[0].shape[0],dianum))
        new_tl = np.zeros((self.force[0].shape[0],dianum))
        new_tr = np.zeros((self.force[0].shape[0],dianum))
        for i in range(len(self.initial_tension)):
            ini = self.initial_tension[i]
            if ini.shape[0]<3:
                continue
            diaix = [int(c[1:]) for c in self.chain[i][1:-1]]
            vec = np.zeros((ini.shape[0]-1,2,2))
            force = np.zeros((ini.shape[0]-1,2))
            for j,d in enumerate(diaix):
                vec[j,:,:] = self.string_vectors[d][i,:,:]
                force[j,:] = self.force[d][i,:]
            lower = [0 for k in range(ini.shape[0])]
            upper = [np.inf for k in range(ini.shape[0])]
            b = scipy.optimize.Bounds(lower,upper)
            result = scipy.optimize.minimize(self._cost,ini,(vec,force),bounds=b)

            out[i] = result.x

        for i in range(len(self.initial_tension)):
            if len(self.chain[i])<3:
                continue
            diaix = [int(c[1:]) for c in self.chain[i][1:-1]]
            for j,d in enumerate(diaix):
                new_tl[i,d] = out[i][j]
                new_tr[i,d] = out[i][j+1]
                new_torque[i,d] = new_tr[i,d]-new_tl[i,d]
            t = out[i]
            vec = np.zeros((len(diaix),2,2))
            for j,d in enumerate(diaix):
                vec[j,:,:] = self.string_vectors[d][i,:,:]
            newforce = self._t2f(t,vec)
            for j,d in enumerate(diaix):
                out_force[d][i,:] = newforce[j]
        return out,out_force,new_tl,new_tr,new_torque

    def _initial_tension(self,chain,force,tl,tr,tl_e,tr_e):
        tension = [np.zeros(0) for i in chain]
        for i,c in enumerate(chain):
            if len(c)>0:
                tension[i] = np.zeros(len(c)-1)
        for i,c in enumerate(chain):
            if len(c)<=1:
                continue
            if len(c)==2:
                tension[i][0]=0.0
                continue
            if len(c)==3:
                diaix = int(c[1][1:])
                tension[i][0] = tl[diaix][i]
                tension[i][-1] = tr[diaix][i]
                if tension[i][0]<0 and tension[i][-1]<0:
                    tension[i][0]=0
                    tension[i][-1]=0
                    continue
                if tension[i][0]<0:
                    tension[i][0]=0
                    tension[i][-1] = self._guess_opposite(
                        self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][0],'r')
                    continue
                if tension[i][-1]<0:
                    tension[i][-1]=0
                    tension[i][0] = self._guess_opposite(
                        self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][-1],'l')
                    continue
                continue

            for j in range(len(c)-3):
                key1 = c[j+1]
                key2 = c[j+2]
                diaix1 = int(key1[1:])
                diaix2 = int(key2[1:])
                tl_s = tl[diaix2][i]
                tr_s = tr[diaix1][i]
                tle_s = tl_e[diaix2][i]
                tre_s = tr_e[diaix1][i]
                ratio = tre_s /(tre_s + tle_s)
                tension[i][j+1] = tr_s*(1-ratio) + tl_s*ratio
                if tension[i][j+1]<0:
                    tension[i][j+1]=0
            diaix = int(c[1][1:])
            tension[i][0] = self._guess_opposite(
                self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][1],'l')
            diaix = int(c[-2][1:])
            tension[i][-1] = self._guess_opposite(
                self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][-2],'r')
        return tension

    def _phi2vector(self,phil,phir):
        vecl = np.stack([np.cos(phil),-np.sin(phil)],axis=1)
        vecr = np.stack([np.cos(phir),-np.sin(phir)],axis=1)
        vec = np.stack((vecl,vecr),axis=1)
        return vec
    
    def _guess_opposite(self,vec,force,tension,toguess):
        if toguess=='l':
            v = vec[0,:]
            v_ori = vec[1,:]
        elif toguess=='r':
            v = vec[1,:]
            v_ori = vec[0,:]
        ten = tension*v_ori
        rest = force - ten
        res = np.sum(rest*v)
        return res
    
    def _t2f(self,tension,vec):
        force = np.zeros((tension.shape[0]-1,2))
        for i in range(force.shape[0]):
            tl = tension[i]
            tr = tension[i+1]
            vecl = vec[i,0,:]
            vecr = vec[i,1,:]
            force[i,:] = tl*vecl + tr*vecr
        return force

    def _cost(self,tension,vec,force):
        current_f = self._t2f(tension,vec)
        residual = force - current_f
        cost = np.sum(residual**2)
        return cost




class TestOptimize():
    def __init__(self):
        from analyzer import Results
        res = Results('./test/pro2')
        res.load()
        chain_diff = res.other.by_key('object_chain')
        framenum = res.other.by_key('frame_number')
        chain = self.tochain(chain_diff,framenum)
        dianum = res.other.by_key('dianum')
        force = [None for i in range(dianum)]
        phi = [None for i in range(dianum)]
        tl = [None for i in range(dianum)]
        tr = [None for i in range(dianum)]
        tl_e = [None for i in range(dianum)]
        tr_e = [None for i in range(dianum)]
        for i in range(dianum):
            key = 'd'+str(i)
            force[i] = res.oned.get_cols([key+'_force_x',key+'_force_y']).values
            phi[i] = res.oned.get_cols([key+'_phi0',key+'_phi1']).values
            tl[i] = res.oned.get_cols([key+'_tension_l']).values
            tr[i] = res.oned.get_cols([key+'_tension_r']).values
            tl_e[i] = res.oned.get_cols([key+'_tl_e']).values
            tr_e[i] = res.oned.get_cols([key+'_tr_e']).values
        calc = TensionOptimization(chain,force,phi,tl,tr,tl_e,tr_e)
        newtension,newforces,newtl,newtr,newtor = calc.get()
        for i in range(dianum):
            key = 'd'+str(i)
            df = pd.DataFrame(newforces[i],columns=[key+'_optforce_x',key+'_optforce_y'])
            res.oned.add_df(df)
            name = [key+'_opttl',key+'_opttr',key+'_opttorque']
            arr = np.stack((newtl[:,i],newtr[:,i],newtor[:,i]),axis=1)
            df = pd.DataFrame(arr,columns=name)
            res.oned.add_df(df)
        res.other.update('tension',newtension)
        res.save()

    def tochain(self,diff,framenum):
        chain = []
        pf = 0
        frames = diff[0]
        chains = diff[1]
        pch = chains[0]
        for frame,ch in zip(frames,chains):
            chain += [pch for i in range(pf,frame)]
            pch = ch
            pf = frame
        ch = diff[1][-1]
        chain += [ch for i in range(pf,framenum)]
        return chain



        
        

        







from guitools import ViewerBase
from visualize import Drawing2,NewDrawing

class ResultsViewerWidget(ViewerBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        # self.drawing = Drawing2(self.pli)
        self.drawing = NewDrawing(self.pli)
        self.pli.showAxis('left',False)
        self.pli.showAxis('bottom',False)

import pyqtgraph.exporters

class ResultsViewerControl(ViewControlBase):
    def __init__(self,direc,loader,df,diff_chain,gravity,dianum,
            stickpos_array,tension):
        super().__init__()
        self.direc = direc
        self.ld = loader
        self.df = df
        self.diff_chain = diff_chain
        self.grav = gravity
        self.massratio = 4
        self.dianum = dianum
        self.stickpos = StickPosition(self.ld.framenum)
        self.stickpos.loadchanges_array(stickpos_array)
        self.window = ResultsViewerWidget()

        self.window.drawing.set_fpos(self.ld.framenum)
        objectkeys = diff_chain[1][0]+diff_chain[2][0]+diff_chain[3][0]
        # self.window.drawing.set_objectkeys(objectkeys)
        posdict = {}
        for key in objectkeys:
            name = [key+'_savgol_x',key+'_savgol_y']
            posdict[key]=df[name].values
        self.window.drawing.set_positions(posdict)
        # self.window.drawing.set_chain(*self._chain())
        # self.window.drawing.set_string(self._chain()[0],posdict)
        self.window.drawing.set_string_tension(self._chain()[0],posdict,tension)
        lacc = self.df[['l_ax','l_ay']].values
        racc = self.df[['r_ax','r_ay']].values
        dforcelist = []
        for i in range(self.dianum):
            key = 'd'+str(i)
            # name = [key+'_force_x',key+'_force_y']
            name = [key+'_optforce_x',key+'_optforce_y']
            dforcelist.append(self.df[name].values)
        lflyframes,rflyframes = self._flyframes()
        # self.window.drawing.set_forces(self.grav,lacc,racc,
        #     dforcelist,lflyframes,rflyframes,self.massratio)
        forcedict = {'l':(lacc-self.grav[0:2])/self.grav[2],'r':(racc-self.grav[0:2])/self.grav[2]}
        for i in range(len(dforcelist)):
            forcedict['d'+str(i)] = dforcelist[i]
        self.window.drawing.set_force(self.grav,forcedict,posdict,lflyframes,rflyframes)

        torquedict = {}
        for i in range(self.dianum):
            key = 'd'+str(i)
            # basename = ['tension_l','tension_r','torque','tl_e','tr_e','dT_e']
            basename = ['opttl','opttr','opttorque','tl_e','tr_e','dT_e']
            name = [key+'_'+n for n in basename]
            tension_l,tension_r,torque,tl_e,tr_e,dT_e =[self.df[n].values for n in name]
            tension = np.stack((tension_l,tension_r),axis=1)
            self.window.plotter.add_diabolo()
            self.window.plotter.set_tension_torque(i,
                np.arange(0,self.ld.framenum),tension,torque,tl_e,tr_e,dT_e)
            torquedict['d'+str(i)] = torque
        self.window.drawing.set_torque(torquedict,posdict)

        thetadict ={} 
        wrapdict = {}
        for i in range(self.dianum):
            key = 'd'+str(i)
            name = [key+'_theta']
            thetadict[key]=np.squeeze(self.df[name].values)
            name = [key+'_wrapstate']
            wrapdict[key]=np.squeeze(self.df[name].values)
        # self.window.drawing.set_wrap(wrapdict,thetadict)
        self.window.drawing.set_wrap(wrapdict,posdict)


        self.window.KeyPressed.connect(self.keyinterp)

        self.fpos=0
        self.change_fpos(0)

    def _chain(self):
        fullchain = [[] for i in range(self.ld.framenum)]
        fullfly = [[] for i in range(self.ld.framenum)]
        frame = self.diff_chain[0]
        chain = self.diff_chain[1]
        flying = self.diff_chain[2]
        for i,(s,e) in enumerate(zip(frame[:],frame[1:]+[self.ld.framenum])):
            fullchain[s:e]=[chain[i] for j in range(s,e)]
            fullfly[s:e]=[flying[i] for j in range(s,e)]
        return fullchain,fullfly
    def _flyframes(self):
        lflys,lflye = self.stickpos.where([3],[1,2,3])
        rflys,rflye= self.stickpos.where([1,2,3],[3])
        lflyframes = np.zeros(self.ld.framenum,dtype=bool)
        rflyframes = np.zeros(self.ld.framenum,dtype=bool)
        for s,e in zip(lflys,lflye):
            lflyframes[s:e]=True
        for s,e in zip(rflys,rflye):
            rflyframes[s:e]=True
        return lflyframes,rflyframes

    def get_window(self):
        return self.window
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        self.window.setcvimage(self.ld.getframe(self.fpos))
        # self.window.drawing.show_fpos(self.fpos)
        # self.window.drawing.show_positions(self.fpos)
        # self.window.drawing.show_forces(self.fpos)
        # self.window.drawing.show_string(self.fpos)
        # self.window.drawing.show_wrap(self.fpos)
        self.window.drawing.update(self.fpos)
        self.window.plotter.show(self.fpos)
        self.window.blockSignals(False)
    def keyinterp(self, key):
        super().keyinterp(key)
        if key == Qt.Key_S:
            self.save_pli()
    def save_pli(self):
        expo = pg.exporters.ImageExporter(self.window.pli)
        # expo.parameters()['width'] = 800
        path = os.path.join(self.direc,'temp_im')
        if not os.path.exists(path):
            os.mkdir(path)
        
        for i in range(self.ld.framenum):
        # for i in range(100):
            print(f'saving:{i}')
            self.change_fpos(i)
            p = os.path.join(path,'img_'+str(i)+'.tif')
            expo.export(p)

class ViewTest():
    def __init__(self):
        impath = './test7/td2.mov'
        ld = Loader(impath)
        dfpath = './test7/testing_6.csv'
        df = pd.read_csv(dfpath,index_col=0)
        txtpath = './test7/testing_5.txt'
        with open(txtpath,mode='r') as f:
            d = json.loads(f.read())
            diff_chain = d['chain_diff']
        stickpos_array = np.array([[127,1,2]])
        grav = (-0.0034861023477654935, 0.24583765706667549, 0.24586237317168205)
        self.c = ResultsViewerControl(ld,
        df,diff_chain,grav,2,stickpos_array)
        self.c.get_window().show()




import sys
from PyQt5.QtWidgets import QApplication
def main():
    # app = QApplication(sys.argv)
    s = TestOptimize()
    # sys.exit(app.exec_())
if __name__ == '__main__':
    main()