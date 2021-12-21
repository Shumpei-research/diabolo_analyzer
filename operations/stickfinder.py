import numpy as np
import pandas as pd
import cv2
import copy
import scipy.ndimage as ndi

from dataclasses import dataclass, asdict,field
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,  
QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import pyqtSignal,QTimer

import sys,os
sys.path.append(os.pardir)
from visualize import DrawPos
from guitools import ImageMaskBaseWidget,ConfigEditor,ViewControlBase



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


class StickFinderBase(QWidget):
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
        self.move(100,100)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l1 = QVBoxLayout()

        self.l0.addLayout(self.l1,3)

        self.l2 = QVBoxLayout()
        self.l0.addLayout(self.l2,1)
        self.configedit = ConfigEditor(self)
        self.l2.addWidget(self.configedit)
        self.finish_button=QPushButton('finish',self)
        self.l2.addWidget(self.finish_button)

        self.subl=[None for i in range(2)]
        self.subl[0]=QHBoxLayout()
        self.subl[1]=QHBoxLayout()
        self.l1.addLayout(self.subl[0])
        self.l1.addLayout(self.subl[1])

        self.imwids = [ImageMaskBaseWidget(self) for i in range(2)]
        self.hsvwids = [ImageMaskBaseWidget(self) for i in range(2)]
        for i in range(2):
            self.subl[i].addWidget(self.imwids[i])
            self.subl[i].addWidget(self.hsvwids[i])

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


class TestStick():
    '''test class'''
    def __init__(self):
        from movieimporter import Loader
        from utilities import StickPosition
        from analyzer import Results
        impath = '../test/td1.mov'
        self.ld = Loader(impath)
        self.stickpos = StickPosition(self.ld.framenum)
        self.res = Results('../test/pro1')
        self.res.load()
        stifra = self.res.other.by_key('stickposition','frame')
        stista = self.res.other.by_key('stickposition','state')
        self.stickpos.loadchanges(stifra,stista)
        name = ['l_x','l_y','l_w','l_h','r_x','r_y','r_w','r_h']
        df = self.res.oned.get_cols(name)
        self.stick = StickFinderControl(self.ld,df,self.stickpos)

        self.stick.finish_signal().connect(self.aa)
    
    def aa(self):
        print('yay')
        print(self.stick.get_df().iloc[600:610,:])
        res = self.stick.get_df()
        newdf = pd.concat((self.df,res),axis=1)
        # newdf.to_csv('./test7/testing_3.csv')

def test():
    app = QApplication([])
    t = TestStick()
    app.exec_()

if __name__=='__main__':
    test()