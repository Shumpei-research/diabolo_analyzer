import os
import sys
import json
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
import tifffile

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, 
QWidget, QVBoxLayout, QTabWidget,
QMenuBar, QAction)
from PyQt5.QtCore import pyqtSignal

from movieimporter import Loader
from visualize import ViewerSet
from operation_base import Operation, StaticOperation
from basicinfo_operation import BasicInfoOperation
from tracking_operation import TrackingOperation, FallTrackingOperation
from circle_operation import CircleOperation, FallCircleOperation
from gravity_operation import GravityOperation
from stickdetection_operation import StickDetectionOperation
from smoothing_operation import SmoothingOperation



class ResultUnitBase(ABC):
    ext = ''
    def __init__(self,path=None,directory=None,name=None):
        self.data = None
        if path is not None:
            self.path = path
        elif not (directory is None or name is None):
            self.path = os.path.join(directory,(name+self.ext))
        else:
            raise AttributeError('path or directory+name must be given.')
    def set(self,data):
        self.data = data
    def get(self):
        return self.data
    def exists(self):
        return os.path.exists(self.path)
    def hasdata(self):
        return not self.data is None
    def load(self):
        if not self.exists():
            raise FileNotFoundError(f'{self.path} does not exist to be loaded.')
        self._rdmain(self.path)
    def save(self):
        if not self.hasdata():
            raise ValueError('No data to save.')
        self._assertdata()
        self._svmain(self.path)
    @abstractmethod
    def _assertdata(self):
        pass
    @abstractmethod
    def _rdmain(self,path):
        pass
    @abstractmethod
    def _svmain(self,path):
        pass

class ResultArray(ResultUnitBase):
    ext = '.npy'
    def _assertdata(self):
        if not isinstance(self.data,np.ndarray):
            raise ValueError(f'data type is not np.ndarray: {self.data}')
    def _svmain(self,path):
        np.save(path,self.data)
    def _rdmain(self,path):
        self.data = np.load(path)
    def add_list(self,data_list:list):
        '''TYX'''
        self.data = np.stack(data_list,axis=0)

class ResultDF(ResultUnitBase):
    ext = '.json'
    def _assertdata(self):
        if not isinstance(self.data,pd.DataFrame):
            raise ValueError(f'data type is not pd.DataFrame: {self.data}')
    def _svmain(self,path):
        self.data.to_json(path)
    def _rdmain(self,path):
        self.data = pd.read_json(path)
    def add_df(self,df:pd.DataFrame):
        '''Append df to self.data. If column(s) exist already, update them.'''
        if not self.hasdata():
            self.data = df
            return
        if df.shape[0] != self.data.shape[0]:
            raise ValueError('row number does not match')
        name = df.columns
        ix = np.isin(name,self.data.columns)
        self.data.update(df.loc[:,ix])
        self.data = pd.concat((self.data,df.loc[:,~ix]),axis=1)

class ResultDict(ResultUnitBase):
    ext = '.json'
    def _assertdata(self):
        if not isinstance(self.data,dict):
            raise ValueError(f'data type is not dict: {self.data}')
    def _rdmain(self,path):
        with open(path,mode='r') as f:
            txt = f.read()
        self.data = json.loads(txt)
    def _svmain(self,path):
        txt = json.dumps(self.data,indent=4)
        with open(path,mode='w') as f:
            f.write(txt)
    def update(self,input_dict:dict):
        '''Append input_dict to self.data. If key(s) exist already, update them.'''
        if not self.hasdata():
            self.data = {}
        self.data.update(input_dict)


class Results():
    def __init__(self,directory):
        self.directory = directory
        self.units = {}

        self._add('log',ResultDict)
        self._add('fall_tracking',ResultDF)
        self._add('fall_circle',ResultDF)
        self._add('gravity',ResultDict)
        self._add('basics',ResultDict)
        self._add('tracking',ResultDF)
        self._add('circle',ResultDF)
        self._add('stick',ResultDF)
        self._add('left_stick',ResultArray)
        self._add('right_stick',ResultArray)
        self._add('smoothened',ResultDF)

    def _add(self,name:str,unitcls):
        self.units[name] = unitcls(directory=self.directory,name=name)
    
    def get_unit(self,name):
        if not name in self.units.keys():
            raise KeyError(f'{name} is not defined.')
        return self.units[name]

    def load(self):
        for unit in self.units.values():
            if unit.exists():
                unit.load()

    def save(self):
        for unit in self.units.values():
            if unit.hasdata():
                unit.save()




class OperationDependence():
    '''Utility class for OperationManager'''
    def __init__(self):
        self.names = []
        self.deps = {}
        self.isdone = {}

        self.update_index = {} # str:int. can be used for tracking when each item is checked. -1:undone, 0~:latest
    def export(self):
        d = {}
        d['names'] = self.names
        d['deps'] = self.deps
        d['isdone'] = self.isdone
        d['update_index'] = self.update_index
        return d
    def load(self,datadict):
        for n in self.names:
            if not n in datadict['names']:
                continue
            if  self.deps[n] != datadict['deps'][n]:
                raise ValueError('operation dependence is different. could not load.')
            self.isdone[n] = datadict['isdone'][n]
            self.update_index[n] = datadict['update_index'][n]
    
    def _check_index(self,key):
        new = {}
        for k,i in self.update_index.items():
            if i==-1:
                new[k] = -1
            else:
                new[k] = 1+i
        self.update_index = new
        self.update_index[key]=0

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
        self.update_index[new] = -1
        if deps is None:
            self.deps[new] = []
            return
        if isinstance(deps,str):
            deps = [deps,]
        self.deps[new] = deps

    def unregister(self,name):
        '''unregister the operation from this instance'''
        if name not in self.names:
            raise ValueError(f'{name} not existing')
        self.names.remove(name)
        self.deps.pop(name)
        self.isdone.pop(name)
    def hasthis(self,name):
        '''returns True is name exists'''
        return name in self.names
    def check(self,name):
        '''set operation(name) as done'''
        if name not in self.names:
            raise ValueError(f'{name} not existing')
        self.isdone[name] = True
        self._check_index(name)
    def uncheck(self,name):
        '''set operation(name) as not done'''
        if name not in self.names:
            raise ValueError(f'{name} not existing')
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
    def get_update_index(self):
        return self.update_index






class OperationTabWidget(QWidget):
    actionTriggered = pyqtSignal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.resize(300,600)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        self.menu = QMenuBar(self)
        self.l0.addWidget(self.menu)
        self.tab = QTabWidget(self)


        self.empty_wid = QWidget()
        self.subwids = {}
        self.tab.addTab(self.empty_wid,'empty')
        self.tab.setCurrentWidget(self.empty_wid)
        self.l0.addWidget(self.tab)

        self.opemenu = self.menu.addMenu('operation')
        self.ope_actions = {}

    def add_operation(self,name,wid):
        newaction  = QAction(name,self)
        self.opemenu.addAction(newaction)
        self.ope_actions[name]=newaction
        newaction.triggered.connect(lambda: self.triggered(name))
        newaction.setDisabled(True)

        self.tab.addTab(wid,name)
        self.subwids[name] = wid
        self.set_empty()

    def enable_operations(self,namelist):
        for n in namelist:
            self.ope_actions[n].setEnabled(True)

    def disable_operations(self,namelist):
        for n in namelist:
            self.ope_actions[n].setDisabled(True)
    
    def triggered(self,name):
        '''operation action triggered'''
        self.tab.setCurrentWidget(self.subwids[name])
        self.actionTriggered.emit(name)
    def set_empty(self):
        self.tab.setCurrentWidget(self.empty_wid)








class OperationManager():
    def __init__(self,res,ld,fall_ld):
        self.res = res
        self.ld = ld
        self.fall_ld = fall_ld

        self.operations = {}
        self.current_operation = None

        self.ope_dep = OperationDependence()
        self.wid = OperationTabWidget()
        self.viewerset = ViewerSet()

        self.wid.actionTriggered.connect(self.run)

        self.add_operation('FallTracking',FallTrackingOperation(self.res,self.fall_ld))
        self.add_operation('FallCircle',FallCircleOperation(self.res,self.fall_ld),'FallTracking')
        self.add_operation('GravityFit',GravityOperation(self.res,self.fall_ld),'FallCircle')
        self.add_operation('BasicInfo',BasicInfoOperation(self.res,self.ld))
        self.add_operation('Tracking',TrackingOperation(self.res,self.ld),'BasicInfo')
        self.add_operation('Circle',CircleOperation(self.res,self.ld),'Tracking')
        self.add_operation('StickDetection',StickDetectionOperation(self.res,self.ld),'Tracking')
        self.add_operation('Smoothing',SmoothingOperation(self.res,self.ld),['Circle','StickDetection'])

        self.wid.enable_operations(self.ope_dep.get_eligible())

    def get_widget(self):
        return self.wid
    def get_viewerwrapper_widget(self):
        return self.viewerset.get_widget()

    def run(self,key):
        if key in self.ope_dep.get_noneligible():
            raise ValueError(f'{key}: dependent operations are not satisfied.')
        self.current_operation = key
        ope = self.operations[key]
        if isinstance(ope,Operation):
            ope.viewer_setting(self.viewerset)
            ope.finish_signal().connect(self.finish_current)
        ope.run()
        if isinstance(ope,StaticOperation):
            self.finish_current()

    def add_operation(self,name,ope:Operation,deps=None):
        self.operations[name] =ope
        self.ope_dep.register(name,deps)
        self.wid.add_operation(name,ope.get_widget())
    
    def add_static_operation(self,name,ope:StaticOperation,deps=None):
        self.operations[name] =ope
        self.ope_dep.register(name,deps)
        self.wid.add_operation(name)
    
    def finish_current(self):
        ope = self.operations[self.current_operation]
        ope.finish_signal().disconnect(self.finish_current)
        ope.post_finish()
        self.ope_dep.check(self.current_operation)
        self.current_operation = None
        self.wid.set_empty()
        self.wid.enable_operations(self.ope_dep.get_eligible())
        self.wid.enable_operations(self.ope_dep.get_done())
    
    def save_state(self):
        unit = self.res.get_unit('log')
        dep = self.ope_dep.export()
        unit.update({'operation_dependence':dep})

    def load_state(self):
        unit = self.res.get_unit('log')
        dep = unit.get()['operation_dependence']
        self.ope_dep.load(dep)
        self.wid.disable_operations(self.ope_dep.get_noneligible())
        self.wid.enable_operations(self.ope_dep.get_eligible())
        self.wid.enable_operations(self.ope_dep.get_done())


class MainWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.resize(200,50)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        self.menu = QMenuBar(self)
        self.l0.addWidget(self.menu)

        self.saveAction = QAction("save", self)
        self.loadAction = QAction("load", self)
        filemenu = self.menu.addMenu("file")
        filemenu.addAction(self.saveAction)
        filemenu.addAction(self.loadAction)
    def save_signal(self):
        return self.saveAction.triggered
    def load_signal(self):
        return self.loadAction.triggered
    

class MainControl():
    def __init__(self,direc,impath,fallpath):
        self.direc = direc
        self.res = Results(direc)
        self.ld = Loader(impath)
        self.fld = Loader(fallpath)
        self.ope = OperationManager(self.res,self.ld,self.fld)
        self.wid = MainWidget()

        self.wid.save_signal().connect(self.save)
        self.wid.load_signal().connect(self.load)

    def get_widgets(self):
        return self.wid, self.ope.get_widget(), self.ope.get_viewerwrapper_widget()
    def show_widgets(self):
        m,o,v = self.get_widgets()
        m.move(100,100)
        o.move(100,220)
        v.move(450,100)
        m.show()
        o.show()
        v.show()

    def save(self):
        self.ope.save_state()
        self.res.save()
    def load(self):
        self.res.load()
        self.ope.load_state()






def maindo(impath,fallimpath,direc):
    app = QApplication(sys.argv)

    if not os.path.exists(direc):
        os.mkdir(direc)

    m = MainControl(direc,impath,fallimpath)
    m.show_widgets()

    sys.exit(app.exec_())

def test():
    impath = os.path.join('.','test','td1.mov')
    fallimpath = os.path.join('.','test','td_fall.mov')
    direc = os.path.join('.','test','protest')
    maindo(impath,fallimpath,direc)

def main():
    impath = sys.argv[1]
    fallimpath = sys.argv[2]
    direc = sys.argv[3]
    maindo(impath,fallimpath,direc)

if __name__ == '__main__':
    test()