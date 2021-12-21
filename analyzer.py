import os
import sys
import json
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from json import JSONEncoder

from utilities import StickPosition
from movieimporter import Loader
from viewer import ResultsViewerControl

from operations.acc import FitPlotControl
from operations.circle import CircleFitControl
from operations.chain import ChainAssignControl
from operations.force import ForceCalcControl
from operations.gravity import GravityFitControl
from operations.stickfinder import StickFinderControl
from operations.stickpos import StickPositionEditorControl
from operations.tension import TensionOptimization
from operations.tracking import TrackerMain
from operations.BGcorrection import BackgroundCorrection,MaskMaker

class DataUnitBase(ABC):
    def __init__(self,name):
        self.data = None
        self.setname(name)
    def setname(self,name:str):
        self.name=name
    def set(self,data):
        self.data = data
    def get(self):
        return self.data
    def load(self,direc):
        path = os.path.join(direc,self.name)
        if os.path.exists(path):
            self._rdmain(path)
    def save(self,direc):
        if self.data is None:
            return
        path = os.path.join(direc,self.name)
        self._svmain(path)
    @abstractmethod
    def _rdmain(self,path):
        pass
    @abstractmethod
    def _svmain(self,path):
        pass



class ResultsOneD(DataUnitBase):
    def _svmain(self,path):
        self.data.to_json(path)
    def _rdmain(self,path):
        self.data = pd.read_json(path)
    def add_df(self,df:pd.DataFrame):
        if self.data is None:
            self.data = df
            return
        if df.shape[0] != self.data.shape[0]:
            return
        name = df.columns
        ix = np.isin(name,self.data.columns)
        self.data.update(df.loc[:,ix])
        self.data = pd.concat((self.data,df.loc[:,~ix]),axis=1)
    def add_array_name(self,arr,name):
        if arr.shape[1] != len(name):
            return
        df = pd.DataFrame(arr,columns=name)
        self.add_df(df)
    def exists(self,name):
        if isinstance(name,str):
            return name in self.data.columns
        for n in name:
            if n not in self.data.columns:
                return False
        return True
    def get_cols(self,name):
        if not self.exists(name):
            return
        return self.data[name]

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return JSONEncoder.default(self, obj)

class ResultsDict(DataUnitBase):
    def _rdmain(self,path):
        with open(path,mode='r') as f:
            txt = f.read()
        self.data = json.loads(txt)
    def _svmain(self,path):
        txt = json.dumps(self.data,indent=4,cls=NumpyEncoder)
        with open(path,mode='w') as f:
            f.write(txt)
    def exists(self,*key):
        target = self.data
        for k in key:
            if k not in target.keys():
                return False
            target = target[k]
        return True
    def by_key(self,*key):
        if not self.exists(*key):
            return
        target = self.data
        for k in key:
            target = target[k]
        return target
    def update(self,key,val):
        if self.data is None:
            self.data = {}
        self.data.update({key:val})


class Results():
    def __init__(self,direc):
        self.direc = direc
        self.oned= ResultsOneD('frame_result_1d.txt')
        self.f_oned = ResultsOneD('fall_oned.txt')
        self.other = ResultsDict('result_others.txt')
        self.log = ResultsDict('log.txt')
    def load(self):
        self.oned.load(self.direc)
        self.other.load(self.direc)
        self.f_oned.load(self.direc)
        self.log.load(self.direc)
    def save(self):
        self.oned.save(self.direc)
        self.other.save(self.direc)
        self.f_oned.save(self.direc)
        self.log.save(self.direc)

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, 
QWidget, QVBoxLayout, QTabWidget,
QMenuBar, QAction)


class Operation(ABC):
    def __init__(self,res,ld):
        '''Results and Loader objects will be registered for .finish() method.'''
        self.res = res
        self.ld = ld
    @abstractmethod
    def isinteractive(self):
        '''returns True if this operation is interactive'''
        return True
    @abstractmethod
    def getname(self):
        '''returns this operation's name'''
        return ''
    @abstractmethod
    def do(self):
        '''perform calculation/interactive operation'''
        pass
    @abstractmethod
    def finish(self):
        '''will be called after finishing operation to take out data.'''
        pass
    @abstractmethod
    def fin_signal(self):
        '''returns pyqt signal that will be emited upon finish'''
        pass
    @abstractmethod
    def get_widget(self):
        '''returns QWidget for this operation'''
        pass


class OperationDependence():
    def __init__(self):
        self.names = []
        self.deps = {}
        self.isdone = {}
    def export(self):
        d = {}
        d['names'] = self.names
        d['deps'] = self.deps
        d['isdone'] = self.isdone
        return d
    def load(self,datadict):
        self.names = datadict['names']
        self.deps = datadict['deps']
        self.isdone = datadict['isdone']

    def register(self,new:str,deps:list=None):
        '''register new operation that depends on deps'''
        if new in self.names:
            raise ValueError(f'{new} already exists')
        if new in self.deps.keys():
            raise ValueError(f'{new} already exists in deps dictionary (unexpected)')
        if new in self.isdone.keys():
            raise ValueError(f'{new} already exists in isdone dictionary (unexpected)')

        self.names.append(new)
        self.isdone[new] = False
        if deps is None:
            self.deps[new] = []
            return
        if isinstance(deps,str):
            deps = [deps,]
        self.deps[new] = deps

    def unregister(self,name):
        '''unregister the operation from this instance'''
        if name not in self.names:
            raise ValueError(f'{name} not exists')
        self.names.remove(name)
        self.deps.pop(name)
        self.isdone.pop(name)
    def hasthis(self,name):
        '''returns True is name exists'''
        return name in self.names
    def check(self,name):
        '''set operation(name) as done'''
        if name not in self.names:
            raise ValueError(f'{name} not exists')
        self.isdone[name] = True
    def uncheck(self,name):
        '''set operation(name) as not done'''
        if name not in self.names:
            raise ValueError(f'{name} not exists')
        self.isdone[name] = False
    def get_eligible(self):
        '''returns list[name] of performable operations (not done)'''
        eligibles = []
        for n in self.names:
            if self.isdone[n]:
                continue
            n_deps = self.deps[n]
            if len(n_deps) == 0:
                eligibles.append(n)
                continue
            status = [self.isdone[d] for d in n_deps]
            if all(status):
                eligibles.append(n)
        return eligibles
    def get_done(self):
        '''returns list[name] of done operations'''
        dones = [n for n in self.names if self.isdone[n]]
        return dones
    def get_noneligible(self):
        '''returns list[name] of not done and noneligible operations'''
        eligibles = self.get_eligible()
        noneligibles = []
        for n in self.names:
            if self.isdone[n]:
                continue
            if not n in eligibles:
                noneligibles.append(n)
        return noneligibles


class FallTrack(Operation):
    def __init__(self,res,ld):
        self.res = res
        self.ld = ld
    def isinteractive(self):
        return True
    def getname(self):
        return 'FallTrack'
    def do(self):
        self.fbc = BackgroundCorrection(self.ld)
        self.fbc.calc()
        self.fmm = MaskMaker(self.ld,self.fbc.get())
        self.fmm.calc()
        self.ftr = TrackerMain(self.fmm.get())
    def finish(self):
        df = self.ftr.get_df()
        name = ['d0_'+n for n in ['x','y','w','h']]
        df = df[name]
        self.res.f_oned.add_df(df)
    def fin_signal(self):
        return self.ftr.finish_signal()
    def get_widget(self):
        return self.ftr.get_window()

class FallCircle(Operation):
    def __init__(self,res,ld):
        self.res = res
        self.ld = ld
    def isinteractive(self):
        return True
    def getname(self):
        return 'FallCircle'
    def do(self):
        df = self.res.f_oned.get()
        self.fci = CircleFitControl(self.ld,df)
    def finish(self):
        df = self.fci.get_df()
        self.res.f_oned.add_df(df)
    def fin_signal(self):
        return self.fci.finish_signal()
    def get_widget(self):
        return self.fci.get_window()

class GravityFit(Operation):
    def __init__(self,res,ld):
        self.res = res
        self.ld = ld
    def isinteractive(self):
        return True
    def getname(self):
        return 'GravityFit'
    def do(self):
        name = ['d0'+n for n in ['c_x','c_y']]
        pos = self.res.f_oned.get_cols(name).values
        self.gra = GravityFitControl(self.ld,pos)
    def finish(self):
        gravity = self.gra.get_g()
        fall_frame = self.gra.get_frames()
        self.res.other.update('gravity',gravity)
        self.res.other.update('fall_frame',fall_frame)
    def fin_signal(self):
        self.gra.finish_signal()
    def get_widget(self):
        return self.gra.get_window()

class OperationWindow(QWidget):
    def __init__(self,res,ld,fld,parent=None):
        super().__init__(parent)
        self.resize(1200,800)
        self.res = res
        self.ld = ld
        self.fld = fld

        self.operations = []
        self.ope_actions = []
        self.ope_dep = OperationDependence()
        self.ope_names = []

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        self.menu = QMenuBar(self)
        self.l0.addWidget(self.menu)

        self.saveAction = QAction("save", self)
        self.loadAction = QAction("load", self)
        filemenu = self.menu.addMenu("file")
        filemenu.addAction(self.saveAction)
        filemenu.addAction(self.loadAction)
        self.saveAction.triggered.connect(self.save)
        self.loadAction.triggered.connect(self.load)

        self.opemenu = self.menu.addMenu('operation')
        self.l1 = QHBoxLayout()
        self.l0.addLayout(self.l1)
        self.currentWidget = None

        self._add(FallTrack(self.res,self.fld))
        self._add(FallCircle(self.res,self.fld),['FallTrack',])
        self._add(GravityFit(self.res,self.fld),['FallCircle',])

        self.ope_actions[0].setEnabled(True)

    def _add(self,ope:Operation,deps=None):
        index = len(self.operations)
        name = ope.getname()

        self.operations.append(ope)
        self.ope_names.append(name)
        self.ope_dep.register(name,deps)

        newaction = QAction(name)
        newaction.setDisabled(True)
        self.ope_actions.append(newaction)
        self.opemenu.addAction(newaction)
        def nonint():
            ope.do()
            self.finishstep(index)
        def interactive():
            ope.do()
            if self.currentWidget is not None:
                self.l1.removeWidget(self.currentWidget)
                self.currentWidget.close()
            self.currentWidget = ope.get_widget()
            self.l1.addWidget(self.currentWidget)
            ope.fin_signal().connect(lambda: self.finishstep(index))
        if ope.isinteractive():
            newaction.triggered.connect(interactive)
        else:
            newaction.triggered.connect(nonint)
    
    def finishstep(self,ix):
        ope = self.operations[ix]
        ope.finish()
        self.ope_dep.check(ope.getname())
        for opename in self.ope_dep.get_eligible():
            i = self.ope_names.index(opename)
            self.ope_actions[i].setEnabled(True)
    
    def save(self):
        self.res.log.update('operation_log',self.ope_dep.export())
        self.res.save()
    def load(self):
        self.res.load()
        status = self.res.log.by_key('operation_log')
        self.ope_dep.load(status)
        for opename in self.ope_dep.get_eligible():
            i = self.ope_names.index(opename)
            self.ope_actions[i].setEnabled(True)
        for opename in self.ope_dep.get_done():
            i = self.ope_names.index(opename)
            self.ope_actions[i].setEnabled(True)
        


class MainControl():
    def __init__(self,direc,impath,fallpath):
        self.direc = direc
        self.res = Results(direc)
        self.ld = Loader(impath)
        self.fld = Loader(fallpath)
        self.wid = OperationWindow(self.res,self.ld,self.fld)
        self.wid.show()


class MainTabWindowBase(QWidget):
    '''depreciated'''
    def __init__(self,parent=None):
        super().__init__(parent)
        self.tab = QTabWidget(self)
        self.menu = QMenuBar(self)
        self.metalayout = QVBoxLayout()
        self.setLayout(self.metalayout)
        self.metalayout.addWidget(self.menu)
        self.metalayout.addWidget(self.tab)

        self.saveAction = QAction("save", self)
        self.loadAction = QAction("load", self)
        self.beginAction = QAction("begin", self)
        actionmenu = self.menu.addMenu("action")
        actionmenu.addAction(self.saveAction)
        actionmenu.addAction(self.loadAction)
        actionmenu.addAction(self.beginAction)

class MainTabWindow(MainTabWindowBase):
    '''depreciated'''
    def __init__(self,parent=None):
        super().__init__(parent)
        self.tab.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.resize(1200,800)

    def addTab(self,wid,name,ix):
        if ix>=self.tab.count():
            self.tab.addTab(wid,name)
            self.tab.setCurrentWidget(wid)
            return
        self.tab.removeTab(ix)
        self.tab.insertTab(ix,wid,name)
        self.tab.setCurrentWidget(wid)


# class DoneFlag():
#     def __init__(self,logres:ResultsDict):
#         self.ix = -1
#         self.logres=logres
#     def get(self):
#         return self.ix
#     def update(self,ix):
#         if self.ix<ix:
#             self.ix = ix
#             self.logres.update('done',self.ix)

class MainTabControl():
    '''depreciated'''
    def __init__(self,impath,fallimpath,direc):
        self.window = MainTabWindow()
        self.direc = direc
        self.res = Results(direc)
        self.done = DoneFlag(self.res.log)
        self.ld = Loader(impath)
        self.fld = Loader(fallimpath)

        self.res.log.update('fall_moviefile',fallimpath)
        self.res.log.update('moviefile',impath)
        self.res.other.update('frame_number',self.ld.framenum)
        self.res.other.update('fall_frame_number',self.fld.framenum)

        self.window.saveAction.triggered.connect(self.res.save)
        self.window.loadAction.triggered.connect(self.load)
        self.window.beginAction.triggered.connect(self.fall_tracking)
    def get_window(self):
        return self.window
    def load(self):
        self.res.load()
        self.done.update(self.res.log.by_key('done'))
        i = self.done.get()
        if i==9:
            self.result_view()
            return
        if i>=0:
            self.fall_circle()
        if i>=1:
            self.gravity_fit()
        if i>=2:
            self.tracking()
        if i>=3:
            self.circle_fit()
        if i>=4:
            self.stick_pos()
        if i>=5:
            self.stick_find()
        if i>=6:
            self.savgol()
        if i>=7:
            self.chain_assign()
        if i>=8:
            self.force_calc()
        if i>=9:
            self.result_view()

    def fall_tracking(self):
        self.fbc = BackgroundCorrection(self.fld)
        self.fbc.calc()
        self.fmm = MaskMaker(self.fld,self.fbc.get())
        self.fmm.calc()
        self.ftr = TrackerMain(self.fmm.get())
        self.window.addTab(self.ftr.get_window(),'fall tracker',0)
        self.ftr.finish_signal().connect(self.ft_fin)
    def ft_fin(self):
        self.done.update(0)
        df = self.ftr.get_df()
        name = ['d0_'+n for n in ['x','y','w','h']]
        df = df[name]
        self.res.f_oned.add_df(df)
        self.fall_circle()
    def fall_circle(self):
        df = self.res.f_oned.get()
        self.fci = CircleFitControl(self.fld,df)
        self.window.addTab(self.fci.get_window(),'fall circle',1)
        self.fci.finish_signal().connect(self.fc_fin)
    def fc_fin(self):
        self.done.update(1)
        df = self.fci.get_df()
        self.res.f_oned.add_df(df)
        self.gravity_fit()
    def gravity_fit(self):
        name = ['d0'+n for n in ['c_x','c_y']]
        pos = self.res.f_oned.get_cols(name).values
        self.gra = GravityFitControl(self.fld,pos)
        self.window.addTab(self.gra.get_window(),'gravity fit',2)
        self.gra.finish_signal().connect(self.gra_fin)
    def gra_fin(self):
        self.done.update(2)
        gravity = self.gra.get_g()
        fall_frame = self.gra.get_frames()
        self.res.other.update('gravity',gravity)
        self.res.other.update('fall_frame',fall_frame)
        self.tracking()
    def tracking(self):
        self.bc = BackgroundCorrection(self.ld)
        self.bc.calc()
        self.mm = MaskMaker(self.ld,self.bc.get())
        self.mm.calc()
        self.tra = TrackerMain(self.mm.get())
        self.window.addTab(self.tra.get_window(),'tracker',3)
        self.tra.finish_signal().connect(self.tra_fin)
    def tra_fin(self):
        self.done.update(3)
        df = self.tra.get_df()
        dianum = self.tra.get_dianum()
        self.res.oned.add_df(df)
        self.res.other.update('dianum',dianum)
        self.circle_fit()
    def circle_fit(self):
        df = self.res.oned.get()
        self.cir = CircleFitControl(self.ld,df)
        self.window.addTab(self.cir.get_window(),'circle',4)
        self.cir.finish_signal().connect(self.cir_fin)
    def cir_fin(self):
        self.done.update(4)
        df = self.cir.get_df()
        self.res.oned.add_df(df)
        self.stick_pos()
    def stick_pos(self):
        self.sti = StickPositionEditorControl(self.ld)
        self.window.addTab(self.sti.get_window(),'stipos',5)
        self.sti.finish_signal().connect(self.pos_fin)
    def pos_fin(self):
        self.done.update(5)
        stipos = self.sti.get_stickpos()
        _,stiframe,stistate = stipos.get()
        self.res.other.update('stickposition',{'frame':stiframe,'state':stistate})
        self.stick_find()
    def stick_find(self):
        name = ['l_x','l_y','l_w','l_h','r_x','r_y','r_w','r_h']
        df = self.res.oned.get_cols(name)
        stifra = self.res.other.by_key('stickposition','frame')
        stista = self.res.other.by_key('stickposition','state')
        framenum = self.res.other.by_key('frame_number')
        stipos = StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        self.stif = StickFinderControl(self.ld,df,stipos)
        self.window.addTab(self.stif.get_window(),'stickfind',6)
        self.stif.finish_signal().connect(self.stif_fin)
    def stif_fin(self):
        self.done.update(6)
        df = self.stif.get_df()
        self.res.oned.add_df(df)
        self.savgol()
    def savgol(self):
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
        self.sav = FitPlotControl(lpos,rpos,dposlist)
        self.window.addTab(self.sav.get_window(),'savgol',7)
        self.sav.finish_signal().connect(self.sav_fin)
    def sav_fin(self):
        self.done.update(7)
        df = self.sav.get_df()
        self.res.oned.add_df(df)
        self.chain_assign()
    def chain_assign(self):
        basename = ['_savgol_x','_savgol_y']
        lname = ['l'+n for n in basename]
        lpos = self.res.oned.get_cols(lname).values
        rname = ['r'+n for n in basename]
        rpos = self.res.oned.get_cols(rname).values
        dposlist = []
        dianum = self.res.other.by_key('dianum')
        for i in range(dianum):
            dname = ['d'+str(i)+n for n in basename]
            dpos = self.res.oned.get_cols(dname).values
            dposlist.append(dpos)
        stifra = self.res.other.by_key('stickposition','frame')
        stista = self.res.other.by_key('stickposition','state')
        framenum = self.res.other.by_key('frame_number')
        stipos = StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        self.ca = ChainAssignControl(self.ld,lpos,rpos,dposlist,stipos)
        self.window.addTab(self.ca.get_window(),'chain',8)
        self.ca.finish_signal().connect(self.ca_fin)
    def ca_fin(self):
        self.done.update(8)
        wrapdf,chain_diff,thetaphidf = self.ca.get_results()
        self.res.oned.add_df(wrapdf)
        self.res.oned.add_df(thetaphidf)
        self.res.other.update('object_chain',chain_diff)
        self.force_calc()
    def force_calc(self):
        grav = self.res.other.by_key('gravity')
        dianum = self.res.other.by_key('dianum')
        acc = [0 for i in range(dianum)]
        phi = [0 for i in range(dianum)]
        theta = [0 for i in range(dianum)]
        for i in range(dianum):
            key = 'd'+str(i)
            acc[i] = self.res.oned.get_cols([key+'_ax',key+'_ay']).values
            phi[i] = self.res.oned.get_cols([key+'_phi0',key+'_phi1']).values
            theta[i] = self.res.oned.get_cols([key+'_theta']).values
        self.frc = ForceCalcControl(grav,acc,phi,theta)
        # no interaction
        self.force_fin()
    def force_fin(self):
        res = self.frc.get_df()
        self.res.oned.add_df(res)
        self.tensionopt_calc()
    def tensionopt_calc(self):
        chain_diff = self.res.other.by_key('object_chain')
        framenum = self.res.other.by_key('frame_number')
        chain = self.tochain(chain_diff,framenum)
        dianum = self.res.other.by_key('dianum')
        force = [None for i in range(dianum)]
        phi = [None for i in range(dianum)]
        tl = [None for i in range(dianum)]
        tr = [None for i in range(dianum)]
        tl_e = [None for i in range(dianum)]
        tr_e = [None for i in range(dianum)]
        for i in range(dianum):
            key = 'd'+str(i)
            force[i] = self.res.oned.get_cols([key+'_force_x',key+'_force_y']).values
            phi[i] = self.res.oned.get_cols([key+'_phi0',key+'_phi1']).values
            tl[i] = self.res.oned.get_cols([key+'_tension_l']).values
            tr[i] = self.res.oned.get_cols([key+'_tension_r']).values
            tl_e[i] = self.res.oned.get_cols([key+'_tl_e']).values
            tr_e[i] = self.res.oned.get_cols([key+'_tr_e']).values
        self.tenop = TensionOptimization(chain,force,phi,tl,tr,tl_e,tr_e)
        # no interaction
        self.tension_fin()
    def tension_fin(self):
        newtension,newforces,newtl,newtr,newtor = self.tenop.get()
        dianum = self.res.other.by_key('dianum')
        for i in range(dianum):
            key = 'd'+str(i)
            df = pd.DataFrame(newforces[i],columns=[key+'_optforce_x',key+'_optforce_y'])
            self.res.oned.add_df(df)
            name = [key+'_opttl',key+'_opttr',key+'_opttorque']
            arr = np.stack((newtl[:,i],newtr[:,i],newtor[:,i]),axis=1)
            df = pd.DataFrame(arr,columns=name)
            self.res.oned.add_df(df)
        self.res.other.update('tension',newtension)
        self.result_view()
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

    def result_view(self):
        df = self.res.oned.get()
        chain = self.res.other.by_key('object_chain')
        grav = self.res.other.by_key('gravity')
        dianum = self.res.other.by_key('dianum')
        stifra = self.res.other.by_key('stickposition','frame')
        stista = self.res.other.by_key('stickposition','state')
        framenum = self.res.other.by_key('frame_number')
        tension = self.res.other.by_key('tension')
        stipos = StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        stiposarr = stipos.get_array()
        self.view = ResultsViewerControl(self.direc,self.ld,df,chain,grav,dianum,stiposarr,tension)
        self.window.addTab(self.view.get_window(),'result',9)
        self.done.update(9)


def test():
    app = QApplication(sys.argv)
    impath = './test/td2.mov'
    fallimpath = './test/td_fall.mov'
    direc = './test/protest'
    if not os.path.exists(direc):
        os.mkdir(direc)
    # m = MainTabControl(impath,fallimpath,direc)
    # m.get_window().show()
    m = MainControl(direc,impath,fallimpath)
    sys.exit(app.exec_())

def main():
    app = QApplication(sys.argv)
    impath = sys.argv[1]
    fallimpath = sys.argv[2]
    direc = sys.argv[3]
    if not os.path.exists(direc):
        os.mkdir(direc)
    m = MainTabControl(impath,fallimpath,direc)
    m.get_window().show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    test()