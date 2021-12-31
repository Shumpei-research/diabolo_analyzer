from copy import deepcopy
from abc import ABC, ABCMeta,abstractmethod
import copy

import numpy as np
import cv2

import pyqtgraph as pg
from PyQt5.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QSlider, QSplitter, QVBoxLayout,
QWidget)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QTextEdit,
QWidget, QPushButton,  
QLineEdit,
QVBoxLayout, QHBoxLayout,
QTabWidget,QLabel,QTreeWidget,QTreeWidgetItem,
QTableWidget, QTableWidgetItem, QMenuBar, QAction)
from PyQt5.QtCore import  QRectF, Qt, pyqtSignal,QTimer
from PyQt5.QtGui import QBrush,QColor,QPainter













class TensionTorquePlotter(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l = QHBoxLayout()
        self.setLayout(self.l)
        self.dianum=0
        self.plots=[]
    def add_diabolo(self):
        newitem = TensionPlotterOne()
        self.plots.append(newitem)
        self.l.addWidget(newitem)
        self.dianum += 1
    def del_diabolo(self,ix):
        if ix in range(self.dianum):
            self.l.removeWidget(self.plots[ix])
            self.plots.pop(ix)
            self.dianum -=1
    def set_tension_torque(self,ix,frames,tension,
            torque,tl_e,tr_e,dT_e):
        if ix in range(self.dianum):
            self.plots[ix].set_tension_torque(frames,tension,
                torque,tl_e,tr_e,dT_e)
    def show(self,fpos):
        for p in self.plots:
            p.show_tension_torque(fpos)

class TensionPlotterOne(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l=QVBoxLayout()
        self.setLayout(self.l)
        self.ini_cnfg()
        self.p1 = pg.PlotItem()
        self.p2= pg.PlotItem()
        self.p1wrap = pg.PlotWidget(plotItem=self.p1)
        self.p2wrap = pg.PlotWidget(plotItem=self.p2)
        self.l.addWidget(self.p1wrap)
        self.l.addWidget(self.p2wrap)
        self.ini_drawings()
    def ini_cnfg(self):
        self.error_factor = 0.7
        self.tel_sty = dict(symbol=None,pen={'color':'c','width':2})
        self.tel_sty_u = dict(symbol=None, pen = {'color':(0,255,255,100),'width':0})
        self.tel_sty_d = dict(symbol=None, pen = {'color':(0,255,255,100),'width':0})
        self.tel_sty_f = dict(color=(0,255,255,100))
        self.ter_sty = dict(symbol=None,pen={'color':'m','width':2})
        self.ter_sty_u = dict(symbol=None,pen={'color':(255,0,255,100),'width':0})
        self.ter_sty_d = dict(symbol=None,pen={'color':(255,0,255,100),'width':0}) 
        self.ter_sty_f = dict(color=[255,0,255,100])
        self.to_sty = dict(symbol=None,pen={'color':'w','width':2})
        self.to_sty_u = dict(symbol=None,pen={'color':(255,255,255,100),'width':0})
        self.to_sty_d = dict(symbol=None,pen={'color':(255,255,255,100),'width':0})
        self.to_sty_f = dict(color=[255,255,255,100])
    def ini_drawings(self):
        vec = np.array([[100,200],[100,200]])

        self.plot_tel = self.p1.plot(vec,**self.tel_sty)
        self.plot_ter = self.p1.plot(vec,**self.ter_sty)
        self.plot_to = self.p2.plot(vec,**self.to_sty)
        self.plot_tel_u = self.p1.plot(vec,**self.tel_sty_u)
        self.plot_ter_u = self.p1.plot(vec,**self.ter_sty_u)
        self.plot_to_u = self.p2.plot(vec,**self.to_sty_u)
        self.plot_tel_d = self.p1.plot(vec,**self.tel_sty_d)
        self.plot_ter_d = self.p1.plot(vec,**self.ter_sty_d)
        self.plot_to_d = self.p2.plot(vec,**self.to_sty_d)

        self.plot_tel_f = pg.FillBetweenItem(self.plot_tel_d,self.plot_tel_u)
        self.plot_tel_f.setBrush(**self.tel_sty_f)
        self.plot_ter_f = pg.FillBetweenItem(self.plot_ter_d,self.plot_ter_u)
        self.plot_ter_f.setBrush(**self.ter_sty_f)
        self.plot_to_f = pg.FillBetweenItem(self.plot_to_d,self.plot_to_u)
        self.plot_to_f.setBrush(**self.to_sty_f)

        self.p1.addItem(self.plot_tel_f)
        self.p1.addItem(self.plot_ter_f)
        self.p2.addItem(self.plot_to_f)

        self.p1fpos = self.p1.addLine(x=0,pen=dict(color='b'))
        self.p1zero = self.p1.addLine(y=0,pen=dict(color='w'))
        self.p2fpos = self.p2.addLine(x=0,pen=dict(color='b'))
        self.p2zero = self.p2.addLine(y=0,pen=dict(color='w'))

        self.p1.setLabel('left','tension (mg)')
        self.p1.setLabel('bottom','frame')
        self.p1.setYRange(-5,10)
        self.p1leg = self.p1.addLegend()
        self.p1leg.addItem(self.plot_tel,'left')
        self.p1leg.addItem(self.plot_ter,'right')
        self.p2.setLabel('left','torque (mgR)')
        self.p2.setLabel('bottom','frame')
        self.p2.setYRange(-10,10)

    def set_tension_torque(self,frames,tension,
            torque,tl_e,tr_e,dT_e):
        self.frames = frames
        self.tension = tension
        self.torque = torque
        self.tl_e = tl_e
        self.tr_e = tr_e
        self.dT_e = dT_e

        x = self.frames[:]
        y = self.tension[:,0][:]
        e = self.tl_e[:] * self.error_factor
        self.plot_tel.setData(x,y)
        self.plot_tel_u.setData(x,y+e)
        self.plot_tel_d.setData(x,y-e)
        y = self.tension[:,1][:]
        e = self.tr_e[:] * self.error_factor
        self.plot_ter.setData(x,y)
        self.plot_ter_u.setData(x,y+e)
        self.plot_ter_d.setData(x,y-e)
        y = self.torque[:]
        e = self.dT_e[:] * self.error_factor
        self.plot_to.setData(x,y)
        self.plot_to_u.setData(x,y+e)
        self.plot_to_d.setData(x,y-e)

    def show_tension_torque(self,fpos):
        self.p1fpos.setPos(fpos)
        self.p2fpos.setPos(fpos)

        self.p1.setXRange(fpos-5,fpos+6)
        self.p2.setXRange(fpos-5,fpos+6)




class DrawItem(ABC):
    def __init__(self,plotitem):
        self.pli = plotitem
        self.item = None
        self.defaultsty()
        self.defaultvis()
        self.generate()
    def setsty(self,sty:dict):
        self.sty=sty
        self.pli.removeItem(self.item)
        self.generate()
    def getsty(self):
        return self.sty
    def setvis(self,vis):
        if (not self.vis) and vis:
            self.item.setVisible(True)
            self.vis = vis
        if self.vis and (not vis):
            self.item.setVisible(False)
            self.vis = vis
    def getvis(self):
        return self.vis
    def defaultvis(self):
        self.vis = True
    def clear(self):
        self.pli.removeItem(self.item)
    @abstractmethod
    def defaultsty(self):
        self.sty={}
    @abstractmethod
    def generate(self):
        pass
    @abstractmethod
    def draw(self):
        pass

class DrawingUnitBase(ABC):
    def __init__(self,plotitem):
        self.makeitem(plotitem)
        self.vis = []
        self.setmastervis(True)
    def setvis(self,vis):
        self.vis = vis
    def setmastervis(self,vis):
        self.mastervis = vis
        self.item.setvis(vis)
    def getvis(self):
        return self.mastervis
    def setsty(self,sty):
        self.item.setsty(sty)
    def getsty(self):
        return self.item.getsty()
    @abstractmethod
    def makeitem(self,plotitem):
        self.item = None
    @abstractmethod
    def set(self):
        pass
    @abstractmethod
    def update(self,fpos):
        if not self.mastervis:
            return
        self.item.setvis(self.vis[fpos])
        if not self.vis[fpos]:
            return
        pass

class DrawPos(DrawItem):
    def defaultsty(self):
        self.sty=dict(pxMode=True,pen=None,symbol='o',symbolSize=10,
            symbolPen={'color':'k','width':2},symbolBrush='m')
    def generate(self):
        self.item = self.pli.plot([0],[0],**self.sty)
    def draw(self,pos):
        if not self.vis:
            return
        pos = np.expand_dims(pos,axis=1)
        self.item.setData(pos[0],pos[1])

class DrawPosUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawPos(pli)
    def set(self,posarray):
        self.data = posarray
        self.vis = np.any(posarray!=0,axis=1)
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(self.data[fpos,:])

class DrawCircle(DrawItem):
    def defaultsty(self):
        self.sty=dict(pxMode=False,pen=None,symbol='o',symbolSize=10,
            symbolPen={'color':'c','width':2},symbolBrush=None)
    def generate(self):
        self.item = self.pli.plot([0],[0],**self.sty)
    def draw(self,center,rad):
        if not self.vis:
            return
        self.item.setData([center[0]],[center[1]],symbolSize=rad*2)

class DrawCircleUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawCircle(pli)
    def set(self,center,rad):
        '''center: ndarray[frame,(x,y)],
        rad: ndarray[frame]'''
        self.center = center
        self.rad = rad
        self.vis = rad!=0
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(self.center[fpos,:],self.rad[fpos])

class DrawRectangle(DrawItem):
    def defaultsty(self):
        self.sty = dict(pxMode=True,pen={'color':'c','width':2},
            symbol=None)
    def generate(self):
        self.item = self.pli.plot([0],[0],**self.sty)
    def draw(self,x,y,w,h):
        if not self.vis:
            return
        nodes = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]])
        self.item.setData(nodes[:,0],nodes[:,1])

class DrawRectangleUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawRectangle(pli)
    def set(self,recs):
        self.recs = recs
        self.vis = np.any(recs!=0,axis=1)
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(*self.recs[fpos,:])

class DrawText(DrawItem):
    def defaultsty(self):
        self.sty={
        'main':{'color':'w','anchor':(0.0,1.0),
            'fill':(100,100,100,100)},
        'font':{'pointSize':20}
        }
    def generate(self):
        self.item = pg.TextItem('',**self.sty['main'])
        font = QFont()
        font.setPointSize(self.sty['font']['pointSize'])
        self.item.setFont(font)
        self.pli.addItem(self.item)
    def draw(self,text,pos):
        if not self.vis:
            return
        self.item.setText(text)
        self.item.setPos(pos[0],pos[1])

class DrawLabelUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawText(pli)
    def set(self,posarray,label):
        self.data = posarray
        self.vis = np.any(posarray!=0,axis=1)
        self.label = label
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(self.label,self.data[fpos,:])

class DrawWrapUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawText(pli)
    def set(self,posarray,label,wrapstate):
        self.data = posarray
        self.vis = np.any(posarray!=0,axis=1)
        self.label = label
        self.wrapstate = wrapstate
    def update(self,fpos):
        super().update(fpos)
        txt = f'{self.label}:{self.wrapstate[fpos]}'
        self.item.draw(txt,self.data[fpos,:])

class DrawTextFixedPos(DrawText):
    def defaultsty(self):
        super().defaultsty()
        self.sty.update({'pos':(10,10)})
    def generate(self):
        self.item = pg.TextItem('',**self.sty['main'])
        font = QFont()
        font.setPointSize(self.sty['font']['pointSize'])
        self.item.setFont(font)
        self.item.setPos(*self.sty['pos'])
        self.pli.addItem(self.item)
    def draw(self,txt):
        if not self.vis:
            return
        self.item.setText(txt)

class DrawFrameUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawTextFixedPos(pli)
    def set(self,framenum):
        self.vis = [True for i in range(framenum)]
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(f'frame: {fpos}')

class DrawString(DrawItem):
    def defaultsty(self):
        self.sty = dict(pxMode=False,pen={'color':'c','width':2},
            symbol=None)
    def generate(self):
        self.item = self.pli.plot([0],[0],**self.sty)
    def draw(self,nodes):
        if not self.vis:
            return
        self.item.setData(nodes[0,:],nodes[1,:])
        
class DrawStringUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawString(pli)
    def set(self,posdict,chain):
        self.posdict = posdict
        self.chain = chain
        self.vis = [len(c)>1 for c in chain]
    def update(self,fpos):
        super().update(fpos)
        pos = []
        for key in self.chain[fpos]:
            pos.append(self.posdict[key][fpos,:])
        nodes = np.array(pos).T
        self.item.draw(nodes)

class DrawArrowhead(DrawItem):
    def defaultsty(self):
        self.sty=dict(pxMode=False,pen=None,
            brush=pg.mkColor('r'),tipAngle=70)
    def generate(self):
        self.item =pg.ArrowItem(angle=0,**self.sty)
        self.pli.addItem(self.item)
    def draw(self,pos,angle,headlen):
        if not self.vis:
            return
        self.item.setPos(pos[0],pos[1])
        self.item.setStyle(angle=angle,headLen=headlen)

class DrawDia(DrawItem):
    def defaultsty(self):
        self.sty = {
        'pos': dict(pxMode=True,pen=None,symbol='o',symbolSize=10,
            symbolPen={'color':'k','width':2},symbolBrush='m'),
        'txt': {'main':{'color':'w','anchor':(0.0,0.0),
                'fill':(100,100,100,100)},
            'font':{'pointSize':20}}
        }
    def defaultvis(self):
        self.vis = {'pos':True,'txt':True}
    def setsty(self,sty):
        self.sty = sty
        self.pos.setsty(self.sty['pos'])
        self.txt.setsty(self.sty['txt'])
    def setvis(self,vis):
        self.vis = vis
        self.pos.setvis(vis['pos'])
        self.txt.setvis(vis['txt'])
    def generate(self):
        self.pos = DrawPos(self.pli)
        self.txt = DrawText(self.pli)
        self.setsty(self.sty)
    def draw(self,pos,txt):
        if self.vis['pos']:
            self.pos.draw((pos[0],pos[1]))
        if self.vis['txt']:
            self.txt.draw(txt,pos)
    def clear(self):
        self.pos.clear()
        self.txt.clear()

class DrawArrow(DrawItem):
    def defaultsty(self):
        self.sty={
        'head':dict(pxMode=False,pen=None,brush='r',tipAngle=70),
        'stem':dict(symbol=None,pen={'color':'r','width':3}),
        'factor':30,
        'headlenfactor': 0.3
        }
    def setsty(self,sty):
        self.sty =sty
        self.head.setsty(self.sty['head'])
        self.stem.setsty(self.sty['stem'])
    def defaultvis(self):
        self.vis = True
    def setvis(self,vis):
        self.vis = vis
        self.head.setvis(vis)
        self.stem.setvis(vis)
    def generate(self):
        self.head = DrawArrowhead(self.pli)
        self.stem = DrawString(self.pli)
        self.setsty(self.sty)
    def draw(self,vec,pos):
        factor = self.sty['factor']
        coords = np.array([[pos[0],factor*vec[0]+pos[0]],
            [pos[1],factor*vec[1]+pos[1]]])
        self.stem.draw(coords)
        if self.vis:
            headl = factor*np.linalg.norm(vec)*self.sty['headlenfactor']
            self.head.draw(coords[:,1],self._getangle(coords),headl)
    def _getangle(self,vec):
        x=vec[0,1]-vec[0,0] 
        y=vec[1,1]-vec[1,0] 
        z = complex(x,y)
        arg = np.angle(z,deg=True) + 180
        return arg
    def clear(self):
        self.head.clear()
        self.stem.clear()

class DrawArrowUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawArrow(pli)
    def set(self,posarray,forcearray):
        self.pos = posarray
        self.force = forcearray
        self.vis = np.any(posarray!=0,axis=1)
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(self.force[fpos,:],self.pos[fpos,:])

class DrawColorLine(DrawItem):
    def defaultsty(self):
        self.sty = {'line':dict(pxMode=False,
            symbol=None),'penwid':4}
    def generate(self):
        self.item = self.pli.plot([0],[0],**self.sty['line'])
    def draw(self,nodes,color):
        if not self.vis:
            return
        self.item.setData(nodes[0,:],nodes[1,:],pen={'color':color,'width':self.sty['penwid']})

class DrawColorString(DrawString):
    def defaultsty(self):
        self.sty = {'edge':{'line':dict(pxMode=False,
            symbol=None),'penwid':4},
            'lim':8.0,
            # 'grad':((0,255,255),(255,0,0))}
            'grad':([0.0,0.5,1.0],[(0,0,255),(255,0,0),(255,255,0)])}
    def setsty(self,sty):
        self.sty =sty
        for l in self.lines:
            l.setsty(self.sty['edge'])
    def setvis(self,vis):
        self.vis = vis
        for l in self.lines[:self.visnum]:
            l.setvis(vis)
    def generate(self):
        self.lines = [DrawColorLine(self.pli) for i in range(10)]
        self.visnum=10
        self.setsty(self.sty)
        self.cm = pg.ColorMap(self.sty['grad'][0],self.sty['grad'][1])
        self.lut = self.cm.getLookupTable(nPts=64)
    def draw(self,nodes,vals):
        if not self.vis:
            return
        for i in range(len(vals)):
            l = self.lines[i]
            l.setvis(True)
            edge = nodes[:,i:i+2]
            val = vals[i]
            ratio = val/self.sty['lim']
            if ratio>1:
                ratio=1
            elif ratio<0:
                ratio=0
            color = self.lut[int(np.floor(ratio*63))]
            l.draw(edge,color)

        for l in self.lines[len(vals):]:
            l.setvis(False)
        
        self.visnum = len(vals)

class DrawColorStringUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawColorString(pli)
    def set(self,posdict,chain,tension):
        self.posdict = posdict
        self.chain = chain
        self.tension = tension
        self.vis = [len(c)>1 for c in chain]
    def update(self,fpos):
        super().update(fpos)
        pos = []
        for key in self.chain[fpos]:
            pos.append(self.posdict[key][fpos,:])
        nodes = np.array(pos).T
        vals = self.tension[fpos]
        self.item.draw(nodes,vals)

class DrawArchArrow(DrawItem):
    def defaultsty(self):
        self.sty = {'line':{'line':dict(pxMode=True,
            symbol=None),'penwid':10},
            'lim':10.0,
            'step':0.1,
            'rad':20.0,
            'grad':((100,50,50),(255,0,0),(50,50,100),(0,0,255))}
    def setsty(self,sty):
        self.sty =sty
        self.line.setsty(self.sty['line'])
    def setvis(self,vis):
        self.vis = vis
        self.line.setvis(vis)
    def generate(self):
        self.line = DrawColorLine(self.pli)
        self.setsty(self.sty)
    def draw(self,val,pos):
        if not self.vis:
            return
        if val>self.sty['lim']:
            val = self.sty['lim']
        if val<-self.sty['lim']:
            val = -self.sty['lim']
        angle = 1.95*np.pi*val/self.sty['lim']
        if angle>0:
            anglearr = np.arange(0,angle,self.sty['step'])
            color = (1-angle/(2*np.pi))*np.array(self.sty['grad'][0]) \
                + (angle/(2*np.pi))*np.array(self.sty['grad'][1])
        elif angle<0:
            anglearr = np.arange(0,angle,-self.sty['step'])
            color = (1+angle/(2*np.pi))*np.array(self.sty['grad'][2]) \
                + (-angle/(2*np.pi))*np.array(self.sty['grad'][3])
        relx = self.sty['rad'] * np.sin(anglearr)
        rely = self.sty['rad'] * (-np.cos(anglearr))
        nodes = np.expand_dims(pos,axis=1) + np.stack((relx,rely),axis=0)

        if nodes.shape[1]<2:
            self.setvis(False)
            return
        self.line.draw(nodes,color)

class DrawTorqueUnit(DrawingUnitBase):
    def makeitem(self,pli):
        self.item = DrawArchArrow(pli)
    def set(self,posarray,torque):
        self.pos = posarray
        self.torque = torque
        self.vis = torque!=0
    def update(self,fpos):
        super().update(fpos)
        self.item.draw(self.torque[fpos],self.pos[fpos,:])

class DrawingBase():
    def __init__(self,pli):
        self.pli = pli
        self.units = {}
    def update(self,fpos):
        for unit in self.units.values():
            unit.update(fpos)
    def getsty(self):
        sty = {key:item.getsty() for key,item in self.units.items()}
        return sty
    def getvis(self):
        vis = {key:item.getvis() for key,item in self.units.items()}
        return vis
    def setsty(self,sty):
        for key,item in self.units.items():
            item.setsty(sty[key])
    def setvis(self,vis):
        for key,item in self.units.items():
            item.setmastervis(vis[key])
    def _add(self,key,unitcls):
        '''insanciate DrawingUnit(unitcls) and register if not existing.
        Set mastervis True in either existing or not.'''
        if not key in self.units.keys():
            self.units[key] = unitcls(self.pli)
        self.units[key].setmastervis(True)
    def vis_off(self):
        '''set mastervis False for every units'''
        for item in self.units.values():
            item.setmastervis(False)

class DefaultSty():
    def __init__(self):
        self.g_arrow={
        'head':dict(pxMode=False,pen=None,brush=(50,50,50),tipAngle=70),
        'stem':dict(symbol=None,pen={'color':(50,50,50),'width':3}),
        'factor':30,
        'headlenfactor': 0.3
        }
        self.f_arrow={
        'head':dict(pxMode=False,pen=None,brush='r',tipAngle=70),
        'stem':dict(symbol=None,pen={'color':'r','width':3}),
        'factor':30,
        'headlenfactor': 0.3
        }
        self.t_arrow={
        'head':dict(pxMode=False,pen=None,brush=(150,50,50),tipAngle=70),
        'stem':dict(symbol=None,pen={'color':(150,50,50),'width':3}),
        'factor':30,
        'headlenfactor': 0.3
        }

class Drawing(DrawingBase):
    def __init__(self,pli):
        super().__init__(pli)
        self.defsty = DefaultSty()
    def set_fpos(self,framenum):
        self._add('frame',DrawFrameUnit)
        self.units['frame'].set(framenum)
    def set_positions(self,posdict):
        for key,pos in posdict.items():
            self._add(key,DrawPosUnit)
            self.units[key].set(pos)
            self._add(key+'_label',DrawLabelUnit)
            self.units[key+'_label'].set(pos,key)
            self.units[key+'_label'].setmastervis(False)
    def set_rectangle(self,key,recs):
        key2 = key+'_rec'
        self._add(key2,DrawRectangleUnit)
        self.units[key2].set(recs)
    def set_circle(self,key,center,rad):
        newkey = key+'_circle'
        self._add(newkey,DrawCircleUnit)
        self.units[newkey].set(center,rad)

    def set_string(self,chain,posdict):
        self._add('string',DrawStringUnit)
        self.units['string'].set(posdict,chain)
    def set_string_tension(self,chain,posdict,tension):
        self._add('colorstring',DrawColorStringUnit)
        self.units['colorstring'].set(posdict,chain,tension)
    def set_force(self,gravity,forcedict,posdict,lvis,rvis):
        gvector,gnorm = gravity[0:2],gravity[2]
        g_normalized = np.array(gvector)/gnorm
        fnum = posdict['l'].shape[0]
        g_normalized = np.tile(g_normalized,(fnum,1))
        for key,force in forcedict.items():
            pos = posdict[key]
            self._add(key+'_g',DrawArrowUnit)
            self._add(key+'_f',DrawArrowUnit)
            self._add(key+'_totalf',DrawArrowUnit)

            self.units[key+'_g'].set(pos,g_normalized)
            self.units[key+'_g'].setsty(self.defsty.g_arrow)
            self.units[key+'_f'].set(pos,force)
            self.units[key+'_f'].setsty(self.defsty.f_arrow)
            self.units[key+'_totalf'].set(pos,force+g_normalized)
            self.units[key+'_totalf'].setsty(self.defsty.t_arrow)
            self.units[key+'_totalf'].setmastervis(False)
        
        sty = deepcopy(self.defsty.g_arrow)
        sty['factor']=sty['factor']*0.25
        self.units['l_g'].setsty(sty)
        self.units['r_g'].setsty(sty)
        sty = deepcopy(self.defsty.f_arrow)
        sty['factor']=sty['factor']*0.25
        self.units['l_f'].setsty(sty)
        self.units['r_f'].setsty(sty)
        sty = deepcopy(self.defsty.t_arrow)
        sty['factor']=sty['factor']*0.25
        self.units['l_totalf'].setsty(sty)
        self.units['r_totalf'].setsty(sty)

        self.units['l_g'].setvis(lvis)
        self.units['l_f'].setvis(lvis)
        self.units['l_totalf'].setvis(lvis)
        self.units['l_totalf'].setmastervis(False)
        self.units['r_g'].setvis(rvis)
        self.units['r_f'].setvis(rvis)
        self.units['r_totalf'].setvis(rvis)
        self.units['r_totalf'].setmastervis(False)

    def set_wrap(self,wrapdict,posdict):
        for key,wrap in wrapdict.items():
            pos = posdict[key]
            self._add(key+'_wrap',DrawWrapUnit)
            self.units[key+'_wrap'].set(pos,key,wrap)
            self.units[key+'_label'].setmastervis(False)
            self.units[key+'_wrap'].setmastervis(False)

    def set_torque(self,torque,posdict):
        for key,val in torque.items():
            pos = posdict[key]
            self._add(key+'_torque',DrawTorqueUnit)
            self.units[key+'_torque'].set(pos,val)

















class ImageWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QHBoxLayout()
        self.glw = pg.GraphicsLayoutWidget(self)
        self.setLayout(self.l0)
        self.l0.addWidget(self.glw)
        self.pli = pg.PlotItem()
        self.pli.setAspectLocked()
        self.imi = pg.ImageItem()
        self.pli.addItem(self.imi)
        self.vb = self.pli.getViewBox()
        self.glw.addItem(self.pli)

        self.maskim = pg.ImageItem()
        self.vb.addItem(self.maskim)
        self.maskim.setZValue(10)
        self.maskim.setOpacity(0.5)
        self.maskim.setLookupTable(np.array([[0,0,0],[255,255,0]]))
        self.maskim.setOpts(compositionMode=QPainter.CompositionMode_Plus)

    
    def get_pli(self):
        return self.pli

    def setcvimage(self,im):
        nim = np.swapaxes(im,0,1)
        nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
        self.imi.setImage(nim)
    def setRect(self,*args):
        rect = QRectF(*args)
        self.imi.setRect(rect)
        self.maskim.setRect(rect)
        self.pli.setRange(rect)

    def add_mask(self):
        self.vb.addItem(self.maskim)
    def del_mask(self):
        self.vb.removeItem(self.maskim)
    def setmask(self,mask):
        mask = np.swapaxes(mask,0,1)
        self.maskim.setImage(mask)




class SliderWidget(QWidget):
    '''QSilder plus que functions'''
    PositionChanged=pyqtSignal(int)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        self.slider = QSlider(Qt.Horizontal)
        self.l0.addWidget(self.slider)
        self.que_layout = QHBoxLayout()
        self.l0.addLayout(self.que_layout)
        self.ques = np.array([],dtype=int)
        self.que_buttons = []

        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)

    def set_framenum(self,min:int,max:int):
        '''max should be framenum-1 '''
        self.slider.setMinimum(min)
        self.slider.setMaximum(max)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self._pos_changed)

    def add_que(self,que:int,label:str=''):
        button = QPushButton(str(que)+'\n'+label)
        ind = self.ques.searchsorted(que)
        self.ques = np.insert(self.ques,ind,que)
        self.que_buttons.insert(ind,button)
        self.que_layout.insertWidget(ind,button)
        button.clicked.connect(lambda: self.set_pos(que))
    
    def clear_cues(self):
        for w in self.que_buttons:
            self.que_layout.removeWidget(w)
            w.close()
        self.que_buttons = []
        self.ques = np.array([],dtype=int)
    
    def set_pos(self,fpos):
        self.slider.setValue(fpos)

    def _pos_changed(self):
        pos = self.slider.value()
        self.PositionChanged.emit(pos)
        self.inactive_time()

    def inactive_time(self):
        self.slider.blockSignals(True)
        self._timer.start(20)
    def inactive_end(self):
        self.slider.blockSignals(False)



class RoiTool():
    def __init__(self,pli:pg.PlotItem):
        self.pli = pli
        self.roi = pg.RectROI((0,0),(100,100))
        self.pli.addItem(self.roi)
    def enable_roi(self,flg:bool):
        if flg:
            self.pli.addItem(self.roi)
        else:
            self.pli.removeItem(self.roi)
    def reg_change_signal(self):
        '''returns pyqtSignal pg.RectROI.sigRegionChangeFinished '''
        return self.roi.sigRegionChangeFinished
    def get_rectangle(self):
        '''returns (x,y,w,h)'''
        x,y = self.roi.pos()
        w,h = self.roi.size()
        x,y,w,h = (int(i) for i in (x,y,w,h))
        return (x,y,w,h)
    def set_roipos(self,x,y):
        self.roi.setPos(x,y)


class PropertyTextWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l0 = QVBoxLayout(self)
        self.setLayout(self.l0)

        self.line = QLineEdit(self)
        self.line.setReadOnly(True)
        self.l0.addWidget(self.line)

        self.maintext = ''
        self.bgrtext = ''
    def _show(self):
        self.line.setText(self.maintext+';   '+self.bgrtext)
    def set_text(self,text):
        self.maintext = text
        self._show()
    def add_bgr(self,vals):
        '''utility method to show [B,G,R] values'''
        self.bgrtext = f'B(H):{vals[0]:.0f}, G(S):{vals[1]:.0f}, R(V):{vals[2]:.0f}'
        self._show()


class FrameWidgetBase(QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)

    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()

    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)

class ViewComposite(FrameWidgetBase):
    KeyPressed = pyqtSignal(int)
    def __init__(self,im_wid:ImageWidget,text_wid:PropertyTextWidget,parent=None):
        super().__init__(parent)

        self.im_wid = im_wid
        self.text_wid = text_wid

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.l0.addWidget(self.im_wid)
        self.l0.addWidget(self.text_wid)
    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def add_slider(self,slider:SliderWidget):
        self.slider = slider
        self.l0.addWidget(self.slider)
    def acitvate_slider(self,flag:bool):
        self.slider.setEnabled(flag)
        if flag:
            self.slider.show()
        else:
            self.slider.hide()










class ViewerFrameConfig():
    def __init__(self):
        self.fast_skip = 5
        self.ff_skip = 25

class ViewerFrameBase(ABC):
    def __init__(self):
        self.view_c = ViewerFrameConfig()
        self.partners = []

    @abstractmethod
    def change_fpos(self,new_fpos,from_partner=False):
        if not from_partner:
            for p in self.partners:
                p.change_fpos(new_fpos,from_partner=True)
    @abstractmethod
    def connect_key(self):
        pass
    @abstractmethod
    def disconnect_key(self):
        pass

    def link_frame(self,target):
        '''link fpos with target'''
        newpartners = list(set(self.partners+target.partners+[self,target]))
        self.partners = [p for p in newpartners if p is not self]
        target.partners = [p for p in newpartners if p is not target]
    
    def unlink_frame(self):
        for p in self.partners:
            p.partners.remove(self)
        self.partners = []

    def keyinterp(self,key):
        if key==Qt.Key_L:
            self.forward()
        if key == Qt.Key_Right:
            self.forward()
        if key==Qt.Key_H:
            self.backward()
        if key == Qt.Key_Left:
            self.backward()
        if key == Qt.Key_K:
            self.fastbackward()
        if key == Qt.Key_Up:
            self.fastbackward()
        if key == Qt.Key_J:
            self.fastforward()
        if key == Qt.Key_Down:
            self.fastforward()
        if key == Qt.Key_F:
            self.ffforward()
        if key == Qt.Key_B:
            self.ffbackward()
    def forward(self):
        self.change_fpos(self.fpos+1)
    def backward(self):
        self.change_fpos(self.fpos-1)
    def fastforward(self):
        self.change_fpos(self.fpos+self.view_c.fast_skip)
    def fastbackward(self):
        self.change_fpos(self.fpos-self.view_c.fast_skip)
    def ffforward(self):
        self.change_fpos(self.fpos+self.view_c.ff_skip)
    def ffbackward(self):
        self.change_fpos(self.fpos-self.view_c.ff_skip)



class SingleViewerSetting():
    def __init__(self):
        self.enable_roi = True
        self.show_roi_bgr = True
        self.enable_slider = True

class SingleViewer(ViewerFrameBase):
    def __init__(self):
        super().__init__()

        self.ld = None
        self.setting = SingleViewerSetting()

        self.im_wid = ImageWidget()
        self.text_wid = PropertyTextWidget()
        self.composite = ViewComposite(self.im_wid,self.text_wid)

        self.pli = self.im_wid.get_pli()
        self.drawing = Drawing(self.pli)
        self.roi = RoiTool(self.pli)

        self.roi.reg_change_signal().connect(self.show_roi_bgr)

        self.slider = SliderWidget()
        self.slider.PositionChanged.connect(self.change_fpos)
        self.composite.add_slider(self.slider)

        self.fpos = 0
        self.connect_key()

        self.cropbox = None # for optioal crop image
        self.cropbool = None

        self.masks = None
    
    def set_loader(self,ld):
        self.ld = ld
        self.slider.set_framenum(0,self.ld.getframenum())
    def set_masks(self,masks:list):
        '''list of ndarray'''
        self.masks = masks

    def set_cropbox(self,box,bool_arr):
        '''ndarray[frame,(xywh)]'''
        self.cropbox = box
        self.cropbool = bool_arr
    def remove_cropbox(self):
        self.cropbox = None
        self.cropbool = None
        self.im_wid.setRect() #reset rect

    def connect_key(self):
        self.composite.KeyPressed.connect(self.keyinterp)
    def disconnect_key(self):
        self.composite.KeyPressed.disconnect(self.keyinterp)

    def get_widget(self):
        return self.composite
    def get_setting(self):
        return self.setting
    def get_drawing(self):
        return self.drawing
    def get_roi(self):
        return self.roi
    def get_fpos(self):
        return self.fpos

    def apply_setting(self):
        s = self.setting
        self.roi.enable_roi(s.enable_roi)
        self.composite.acitvate_slider(s.enable_slider)
        
    def show_roi_bgr(self):
        if not (self.setting.enable_roi and self.setting.show_roi_bgr):
            return
        (x,y,w,h) = self.roi.get_rectangle()
        frame = self.ld.getframe(self.fpos)
        crop = frame[y:y+h,x:x+w,:]
        vals = np.mean(crop,axis=(0,1)).tolist()
        self.text_wid.add_bgr(vals)

    def change_fpos(self,new_fpos,from_partner=False):
        '''try to update self.fpos to new_fpos.
        returns True is update was successful.
        self.fpos remains unchanged if unsuccessful.'''
        super().change_fpos(new_fpos,from_partner)
        if not self.ld.hasframe(new_fpos):
            return False
        frame = self.ld.getframe(new_fpos)
        if self.cropbox is None:
            self.im_wid.setcvimage(frame)
            if not self.masks is None:
                self.im_wid.setmask(self.masks[new_fpos])
        elif not self.cropbool[new_fpos]:
            return False
        else:
            x,y,w,h = self.cropbox[new_fpos,:]
            crop = frame[y:y+h,x:x+w,:]
            self.im_wid.setcvimage(crop)
            if not self.masks is None:
                maskcrop = self.masks[new_fpos][y:y+h,x:x+w]
                self.im_wid.setmask(maskcrop)
            self.im_wid.setRect(x,y,w,h)
            self.roi.set_roipos(x,y)
        self.drawing.update(new_fpos)
        self.slider.set_pos(new_fpos)

        self.fpos = new_fpos
        self.text_wid.set_text(f'frame {self.fpos}')
        self.show_roi_bgr()

        return True




class StaticPlotWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        self.p0 = pg.PlotItem()
        self.p0wrap = pg.PlotWidget(plotItem=self.p0)
        self.l0.addWidget(self.p0wrap)

        self.p0.showGrid(x=True,y=True,alpha=0.5)
    
    def get_pli(self):
        '''return pg.PlotItem'''
        return self.p0

class StaticPlotViewer(QWidget):
    def __init__(self):
        self.wid = StaticPlotWidget()
    def get_widget(self):
        return self.wid
    def get_pli(self):
        '''return pg.PlotItem'''
        return self.wid.get_pli()



class FramePlotWidget(FrameWidgetBase):
    KeyPressed = pyqtSignal(int)
    def __init__(self,parent=None):
        super().__init__(parent)

        self.frange = [-10,10]

        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        self.p0 = pg.PlotItem()
        self.p0wrap = pg.PlotWidget(plotItem=self.p0)
        self.l0.addWidget(self.p0wrap)

        self.fposline = self.p0.addLine(x=0,pen=dict(color=(100,100,100),width=1))
        self.p0.setLabel('bottom','frame')

    def get_pli(self):
        '''return pg.PlotItem'''
        return self.p0
    def set_frange(self,offsets:list):
        '''if fpos +- 10, offsets=[-10,10].'''
        self.frange = offsets

    def set_fpos(self,fpos):
        self.p0.setXRange(fpos+self.frange[0],fpos+self.frange[1])
        self.fposline.setPos(fpos)


class FramePlotViewer(ViewerFrameBase):
    def __init__(self):
        super().__init__()
        self.wid = FramePlotWidget()
        self.connect_key()
    
    def connect_key(self):
        self.wid.KeyPressed.connect(self.change_fpos)
    def disconnect_key(self):
        self.wid.KeyPressed.disconnect(self.change_fpos)

    def change_fpos(self,new_fpos,from_partner=False):
        super().change_fpos(new_fpos,from_partner)
        self.wid.set_fpos(new_fpos)







class ViewerSet():
    ViewerDictionary = {
        'single':SingleViewer,
        'fplot':FramePlotViewer,
        'splot':StaticPlotViewer
    }

    def __init__(self):
        super().__init__()
        self.viewers = {}
        self.wid = ViewerSetWidget()
        self.fpos=0
    def get_widget(self):
        return self.wid
    def clear_viewers(self):
        for k,l in self.viewers.items():
            for viewer in l:
                viewer.get_widget().close()
            self.viewers[k] = []
    def generate_viewers(self,order:dict):
        '''order is {key1:number1,key2:number2,...}'''
        self.clear_viewers()
        for key,number in order.items():
            if not key in ViewerSet.ViewerDictionary.keys():
                raise ValueError(f'key {key} is not registered')
            self.viewers[key] = []
            for i in range(number):
                self.viewers[key].append(ViewerSet.ViewerDictionary[key]())
    def get_viewers(self):
        return self.viewers
    def deploy(self,presetkey:str,**kwargs):
        wid_dict = {key:[v.get_widget() for v in l] for key,l in self.viewers.items()}
        self.wid.deploy(wid_dict,presetkey,**kwargs)



class TabPreset(QWidget):
    def __init__(self,widgets:dict,rows:dict,cols:dict,tabnames:dict,order='cols'):
        '''basically grid preset, but change tabs when r/c is filled.'''
        super().__init__(parent=None)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        if len(widgets)>1:
            self.splitter = QSplitter(Qt.Horizontal)
            self.l0.addWidget(self.splitter)
        else:
            # for only one category, splitter is not suitable
            self.splitter = QHBoxLayout()
            self.l0.addLayout(self.splitter)

        for key,wids in widgets.items():
            tab = QTabWidget()
            self.splitter.addWidget(tab)
            nwid = rows[key]*cols[key]
            tabnum = len(wids)//nwid

            if isinstance(order,dict):
                o = order[key]
            elif isinstance(order,str):
                o = order
            else:
                raise TypeError(f'order {order} unexpected type')

            for t in range(tabnum):
                gw = QWidget()
                grid = QGridLayout()
                gw.setLayout(grid)
                tab.addTab(gw,tabnames[key][t])

                for i in range(t*nwid,(t+1)*nwid):
                    if o=='cols':
                        r = i//cols[key]
                        c = i%cols[key]
                    elif o=='rows':
                        r = i%rows[key]
                        c = i//rows[key]
                    else:
                        raise ValueError(f'order {o} unexpected')
                    grid.addWidget(wids[i],r,c)


class GridPreset(QWidget):
    def __init__(self, widgets:dict, rows:dict, cols:dict, order='cols'):
        super().__init__(parent=None)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        if len(widgets)>1:
            self.splitter = QSplitter(Qt.Horizontal)
            self.l0.addWidget(self.splitter)
        else:
            # for only one category, splitter is not suitable
            self.splitter = QHBoxLayout()
            self.l0.addLayout(self.splitter)

        self.l1 = {}
        for key in widgets.keys():
            wid = QFrame(frameShape=QFrame.Panel)
            self.l1[key] = QGridLayout()
            wid.setLayout(self.l1[key])
            self.splitter.addWidget(wid)

        for key,wids in widgets.items():
            if isinstance(order,dict):
                o = order[key]
            elif isinstance(order,str):
                o = order
            else:
                raise TypeError(f'order {order} unexpected type')
            for i,wid in enumerate(wids):
                if o=='cols':
                    r = i//cols[key]
                    c = i%cols[key]
                elif o=='rows':
                    r = i%rows[key]
                    c = i//rows[key]
                else:
                    raise ValueError(f'order {o} unexpected')
                self.l1[key].addWidget(wid,r,c)


class SinglePreset(QWidget):
    def __init__(self,widgets):
        super().__init__(parent=None)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l0.addWidget(widgets['single'][0])


class ViewerSetWidget(QWidget):
    PresetDictionary = {
        'grid':GridPreset,
        'single':SinglePreset,
        'tab':TabPreset
    }
    def __init__(self,parent=None):
        super().__init__(parent)
        self.resize(800,800)

        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        self.current_wid = QWidget()
        self.l0.addWidget(self.current_wid)

    def deploy(self,widgets:dict,presetkey:str,**kwargs):
        self.l0.removeWidget(self.current_wid)
        self.current_wid.close()
        self.current_wid = ViewerSetWidget.PresetDictionary[presetkey](widgets,**kwargs)
        self.l0.addWidget(self.current_wid)