import copy
import numpy as np
from dataclasses import dataclass, asdict, field
import scipy.signal
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,  
QVBoxLayout, QHBoxLayout, QTabWidget)
from PyQt5.QtCore import pyqtSignal

import pyqtgraph as pg

import sys,os
sys.path.append(os.pardir)
from guitools import ConfigEditor

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


class FitPlotBase(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
    def initUI(self):
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        
        self.glw = [None for i in range(6)]
        self.pli = [None for i in range(6)]
        for i in range(6):
            self.glw[i] = pg.GraphicsLayoutWidget(self)
            self.l0.addWidget(self.glw[i])
            self.pli[i] = pg.PlotItem()
            self.glw[i].addItem(self.pli[i])

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

        self.pli[0].setLabel('left','2nd dif')
        self.pli[1].setLabel('left','1st dif')
        self.pli[2].setLabel('left','savgol')
        self.pli[3].setLabel('left','res')
        self.pli[4].setLabel('left','outlier')
        self.pli[5].setLabel('left','outlier res')
        for i in range(1,6):
            self.pli[i].setXLink(self.pli[0].getViewBox())

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
        

class FitPlotWidget(QWidget):
    Updated = pyqtSignal(str,dict)
    def __init__(self,key:str,parent=None):
        self.key = key
        super().__init__(parent)
        self.initUI()
        self.configedit.UpdatedConfig.connect(self.updated)

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
    def updated(self,d):
        self.Updated.emit(self.key,d)


class FitPlotTabs(QTabWidget):
    Finished = pyqtSignal(str)
    Updated = pyqtSignal(str,dict)
    def __init__(self,dianum,parent=None):
        super().__init__(parent)
        self.dianum = dianum
        self.initUI()
        self.left.fin_button.clicked.connect(lambda: self.finished('l'))
        self.left.Updated.connect(self.updated)
        self.right.fin_button.clicked.connect(lambda: self.finished('r'))
        self.right.Updated.connect(self.updated)
        keys = ['d'+str(i) for i in range(self.dianum)]
        for val,key in zip(self.dias,keys):
            val.fin_button.clicked.connect(lambda: self.finished(key))
            val.Updated.connect(self.updated)
    def initUI(self):
        self.left = FitPlotWidget('l')
        self.right = FitPlotWidget('r')
        self.dias = [FitPlotWidget('d'+str(i)) for i in range(self.dianum)]
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
        # for i in range(1):
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
        from analyzer import Results
        self.res = Results('../test/pro2')
        self.res.load()
        basename = ['tip_x','tip_y']
        lname = ['l'+n for n in basename]
        rname = ['r'+n for n in basename]
        lpos = self.res.oned.get_cols(lname).values
        rpos = self.res.oned.get_cols(rname).values
        dbasename = ['c_x','c_y']
        dposlist = []
        dianum = self.res.other.by_key('dianum')
        for i in range(dianum):
            dname = ['d'+str(i)+n for n in dbasename]
            dposlist.append(self.res.oned.get_cols(dname).values)
        
        self.control = FitPlotControl(lpos,rpos,dposlist)
        self.window = self.control.get_window()
        self.window.show()
        self.control.finish_signal().connect(self.finish)
    def finish(self):
        print('a')
        res = self.control.get_df()
        # df = pd.concat((self.df2,self.df1,res),axis=1)
        # df.to_csv('./test7/testing_4.csv')

def test():
    app = QApplication([])
    t = TestFit()
    app.exec_()

if __name__=='__main__':
    test()