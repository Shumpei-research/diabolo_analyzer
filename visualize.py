from copy import deepcopy
from abc import ABC,abstractmethod
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout,
QWidget)
from PyQt5.QtGui import QFont


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


class Drawing():
    '''
    class to add/update drawings (pg.GraphicsItems) to the given canvas (pg.plotItem).
    contains data for all frames.
    update drawings by specifying frame number.
    '''
    def __init__(self,plotitem):
        self.pli = plotitem
        self.item = {}
        self.dianum = 0
        self.mastervis = {}
        self.framevis = {}
    def clear(self):
        for item in self.item.values():
            item.clear()
        self.item={}
    def getsty(self):
        sty = {key:item.getsty() for key,item in self.item.items()}
        return sty
    def setsty(self,sty):
        for key,item in self.item.items():
            item.setsty(sty[key])
    def getvis(self):
        return self.mastervis
    def setvis(self,vis):
        self.mastervis = vis
    def _check(self,key,fp):
        if key not in self.mastervis.keys():
            self.mastervis[key]=self.item[key].getvis()
        m = self.mastervis[key]
        f = self.framevis[key][fp]
        def _rec_and(a,b):
            if isinstance(a,bool):
                return a and b
            if isinstance(a,dict):
                out = {}
                for key in a.keys():
                    out[key]=_rec_and(a[key],b[key])
                return out
        vis = _rec_and(m,f)
        self.item[key].setvis(vis)
    def set_fpos(self,framenum):
        self.item['frame'] = DrawTextFixedPos(self.pli)
        self.framevis['frame'] = [True for i in range(framenum)]
    def show_fpos(self,fp):
        if 'frame' not in self.item.keys():
            return
        self._check('frame',fp)
        self.item['frame'].draw(f'frame: {fp}')
    def set_objectkeys(self,objectkeys):
        self.objectkeys = objectkeys
        self.dianum=0
        for key in objectkeys:
            if key in ['l','r']:
                self.item.update({key:DrawPos(self.pli)})
                self.item.update({key+'_key':DrawText(self.pli)})
            elif key[0]=='d':
                self.item.update({key:DrawPos(self.pli)})
                self.item.update({key+'_key':DrawText(self.pli)})
                self.dianum += 1
    def set_positions(self,posdict):
        '''pos = ndarray(frames,x-y)'''
        self.posdict = posdict
        for key,pos in self.posdict.items():
            self.framevis[key]=list(~np.all(pos==0,axis=1))
            self.framevis[key+'_key']=list(~np.all(pos==0,axis=1))
    def show_positions(self,fp):
        if not hasattr(self,'objectkeys'):
            return
        for key in self.objectkeys:
            dpos = self.posdict[key][fp,:]
            self._check(key,fp)
            self.item[key].draw(dpos)
            self._check(key+'_key',fp)
            self.item[key+'_key'].draw(key,dpos)
    def set_chain(self,chain,flying):
        self.item.update({'string':DrawString(self.pli)})
        self.chain = chain
        self.flying = flying
        self.framevis['string']=[len(c)>1 for c in chain]
    def show_string(self,fp):
        if 'string' not in self.item.keys():
            return
        pos = []
        for key in self.chain[fp]:
            pos.append(self.posdict[key][fp,:])
        nodes = np.array(pos).T
        self._check('string',fp)
        self.item['string'].draw(nodes)
    def set_forces(self,gravity,forcedict):
        ''' gravity: (gx,gy,gnorm)'''
        self.gvector,self.gnorm = gravity[0:2],gravity[2]
        self.forcedict = forcedict
        for key in self.objectkeys:
            ga = DrawArrow(self.pli)
            sty = ga.getsty()
            sty['head']['brush']=(50,50,50)
            sty['stem']['pen']['color']=(50,50,50)
            if key in ['l','r']:
                sty['factor']=7.5
            ga.setsty(sty)
            self.item.update({key+'_g':ga})

            fa = DrawArrow(self.pli)
            sty = fa.getsty()
            if key in ['l','r']:
                sty['factor']=7.5
            fa.setsty(sty)
            self.item.update({key+'_f':fa})

            toa = DrawArrow(self.pli)
            sty = toa.getsty()
            sty['head']['brush']=(150,50,50)
            sty['stem']['pen']['color']=(150,50,50)
            if key in ['l','r']:
                sty['factor']=7.5
            toa.setsty(sty)
            self.item.update({key+'_totalf':toa})

            ix = self.framevis[key]
            self.framevis.update({
                key+'_g': deepcopy(ix),
                key+'_f': deepcopy(ix),
                key+'_totalf': deepcopy(ix)
            })
            self.mastervis[key+'_totalf']=False
            if key in ['l','r']:
                self.mastervis[key+'_g']=False
                self.mastervis[key+'_f']=False
    def show_forces(self,fp):
        g_normalized = np.array(self.gvector)/self.gnorm
        for key in self.objects:
            pos = self.posdict[key][fp,:]
            f = self.forcedict[key][fp,:]
            f_normalized = f
            total = f_normalized + g_normalized
            self._check(key+'_g',fp)
            self._check(key+'_f',fp)
            self._check(key+'_totalf',fp)
            self.item[key+'_g'].draw(g_normalized,pos)
            self.item[key+'_f'].draw(f_normalized,pos)
            self.item[key+'_totalf'].draw(total,pos)
    def set_wrap(self,wrapdict,angledict):
        '''angledict: generalized wrap angle'''
        self.wrapstates = wrapdict
        self.wrapangles = angledict
    def show_wrap(self,fp):
        for i in range(self.dianum):
            key = 'd'+str(i)
            if key not in self.posdict.keys():
                return
            dpos = self.posdict[key][fp,:]
            if not hasattr(self,'wrapstates'):
                return
            if key not in self.wrapstates.keys():
                return
            state = self.wrapstates[key][fp]
            angle = self._contactangle(self.wrapangles[key][fp])
            txt = f'{key}:{state}:{angle:.1f}'
            self._check(key,fp)
            self.item[key+'_key'].draw(txt,dpos)
    def _contactangle(self,generalangle):
        a = generalangle/(2*np.pi)
        if a>0.5:
            return a-0.5
        if a<-0.5:
            return a+0.5
        if -0.5<=a and a<=0.5:
            return 0

class Drawing2(Drawing):
    def set_forces(self,gravity,lacc,racc,dforcelist,
            lflyframes,rflyframes,massratio):
        self.gvector,self.gnorm = gravity[0:2],gravity[2]
        self.lacc = lacc
        self.racc = racc
        self.dforcelist = dforcelist
        for key in self.objectkeys:
            ga = DrawArrow(self.pli)
            sty = ga.getsty()
            sty['head']['brush']=(50,50,50)
            sty['stem']['pen']['color']=(50,50,50)
            if key in ['l','r']:
                sty['factor']=30.0/massratio
            ga.setsty(sty)
            self.item.update({key+'_g':ga})

            fa = DrawArrow(self.pli)
            sty = fa.getsty()
            if key in ['l','r']:
                sty['factor']=30.0/massratio
            fa.setsty(sty)
            self.item.update({key+'_f':fa})

            toa = DrawArrow(self.pli)
            sty = toa.getsty()
            sty['head']['brush']=(150,50,50)
            sty['stem']['pen']['color']=(150,50,50)
            if key in ['l','r']:
                sty['factor']=30.0/massratio
            toa.setsty(sty)
            self.item.update({key+'_totalf':toa})

            ix = self.framevis[key]
            self.framevis.update({
                key+'_g': deepcopy(ix),
                key+'_f': deepcopy(ix),
                key+'_totalf': deepcopy(ix)
            })
            self.mastervis[key+'_totalf']=False
        self.framevis['l_g'] = deepcopy(lflyframes)
        self.framevis['l_f'] = deepcopy(lflyframes)
        self.framevis['l_totalf'] = deepcopy(lflyframes)
        self.framevis['r_g'] = deepcopy(rflyframes)
        self.framevis['r_f'] = deepcopy(rflyframes)
        self.framevis['r_totalf'] = deepcopy(rflyframes)
    def show_forces(self, fp):
        g_normalized = np.array(self.gvector)/self.gnorm
        for key in self.objectkeys:
            pos = self.posdict[key][fp,:]
            if key[0]=='d':
                ix = int(key[1:])
                f = self.dforcelist[ix][fp,:]
                f_normalized = f
                total = f_normalized + g_normalized
            elif key =='l':
                total = self.lacc[fp,:]/self.gnorm
                f_normalized = total - g_normalized
            elif key =='r':
                total = self.racc[fp,:]/self.gnorm
                f_normalized = total - g_normalized
            self._check(key+'_g',fp)
            self._check(key+'_f',fp)
            self._check(key+'_totalf',fp)
            self.item[key+'_g'].draw(g_normalized,pos)
            self.item[key+'_f'].draw(f_normalized,pos)
            self.item[key+'_totalf'].draw(total,pos)



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
            # self.pli.addItem(self.item)
            self.vis = vis
        if self.vis and (not vis):
            self.item.setVisible(False)
            # self.pli.removeItem(self.item)
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

class DrawText(DrawItem):
    def defaultsty(self):
        self.sty={
        'main':{'color':'w','anchor':(0.0,0.0),
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
    def _add(self,key,unit):
        if not key in self.units.keys():
            self.units[key] = unit

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

class NewDrawing(DrawingBase):
    def __init__(self,pli):
        super().__init__(pli)
        self.defsty = DefaultSty()
    def set_fpos(self,framenum):
        self._add('frame',DrawFrameUnit(self.pli))
        self.units['frame'].set(framenum)
    def set_positions(self,posdict):
        for key,pos in posdict.items():
            self._add(key,DrawPosUnit(self.pli))
            self.units[key].set(pos)
            self._add(key+'_label',DrawLabelUnit(self.pli))
            self.units[key+'_label'].set(pos,key)
            self.units[key+'_label'].setmastervis(False)
    def set_string(self,chain,posdict):
        self._add('string',DrawStringUnit(self.pli))
        self.units['string'].set(posdict,chain)
    def set_string_tension(self,chain,posdict,tension):
        self._add('colorstring',DrawColorStringUnit(self.pli))
        self.units['colorstring'].set(posdict,chain,tension)
    def set_force(self,gravity,forcedict,posdict,lvis,rvis):
        gvector,gnorm = gravity[0:2],gravity[2]
        g_normalized = np.array(gvector)/gnorm
        fnum = posdict['l'].shape[0]
        g_normalized = np.tile(g_normalized,(fnum,1))
        for key,force in forcedict.items():
            pos = posdict[key]
            self._add(key+'_g',DrawArrowUnit(self.pli))
            self._add(key+'_f',DrawArrowUnit(self.pli))
            self._add(key+'_totalf',DrawArrowUnit(self.pli))

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
        self.units['l_totalf'].setmastervis(True)
        self.units['l_totalf'].setmastervis(False)
        self.units['r_g'].setvis(rvis)
        self.units['r_f'].setvis(rvis)
        self.units['r_totalf'].setvis(rvis)
        self.units['r_totalf'].setmastervis(True)
        self.units['r_totalf'].setmastervis(False)


    def set_wrap(self,wrapdict,posdict):
        for key,wrap in wrapdict.items():
            pos = posdict[key]
            self._add(key+'_wrap',DrawWrapUnit(self.pli))
            self.units[key+'_wrap'].set(pos,key,wrap)
            self.units[key+'_label'].setmastervis(False)
            self.units[key+'_wrap'].setmastervis(False)
    def set_torque(self,torque,posdict):
        for key,val in torque.items():
            pos = posdict[key]
            self._add(key+'_torque',DrawTorqueUnit(self.pli))
            self.units[key+'_torque'].set(pos,val)
