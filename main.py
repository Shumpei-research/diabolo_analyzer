import os
import sys
import json
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
from json import JSONEncoder

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, 
QWidget, QVBoxLayout, QTabWidget,
QMenuBar, QAction)
from PyQt5.QtCore import pyqtSignal

from movieimporter import Loader



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








class Operation(ABC):
    def __init__(self,res,ld,viewer):
        '''Results and Loader objects will be registered for .finish() method.'''
        self.viewer=viewer
        self.res = res
        self.ld = ld
    @abstractmethod
    def run(self):
        '''perform calculation/interactive operation'''
        pass
    @abstractmethod
    def post_finish(self):
        '''will be called after finishing operation to take out data.'''
        pass
    @abstractmethod
    def finish_signal(self):
        '''returns pyqtSignal that will be emited upon finish'''
        pass
    @abstractmethod
    def get_widget(self):
        '''returns QWidget for this operation'''
        pass
    @abstractmethod
    def viewer_setting(self):
        '''set viewer'''
        pass



class StaticOperation(ABC):
    '''not interactive'''
    def __init__(self,res,ld):
        '''Results and Loader objects will be registered for .finish() method.'''
        self.res = res
        self.ld = ld
    @abstractmethod
    def run(self):
        '''perform calculation'''
        pass
    @abstractmethod
    def post_finish(self):
        '''save etc.'''
        pass




        
class OperationDependence():
    '''Utility class for OperationManager'''
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






class OperationTabWidget(QWidget):
    actionTriggered = pyqtSignal(str)

    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        self.menu = QMenuBar(self)
        self.l0.addWidget(self.menu)

        self.opemenu = self.menu.addMenu('operation')
        self.ope_actions = {}

    def add_operation(self,name):
        newaction  = QAction(name,self)
        self.ope_actions[name]=newaction
        newaction.triggered.connect(lambda: self.triggered(name))
        newaction.setDisabled(True)

    def enable_operations(self,namelist):
        for n in namelist:
            self.ope_actions[n].setEnabled(True)

    def disable_operations(self,namelist):
        for n in namelist:
            self.ope_actions[n].setDisabled(True)
    
    def triggered(self,name):
        '''operation action triggered'''
        self.actionTriggered.emit(name)





class ViewerBase(ABC):
    @abstractmethod
    def __init__(self,ld:Loader):
        self.ld = ld
    @abstractmethod
    def get_widget(self):
        pass






class ViewerWrapperWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.tab = QTabWidget(self)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l0.addWidget(self.tab)

    def add_viewer(self,wid,name):
        self.tab.addTab(wid,name)
    
    def activate_viewer(self,widget):
        self.tab.setCurrentWidget(widget)





class ViewerManager():
    def __init__(self):
        self.viewers = {}
        self.wid = ViewerWrapperWidget()

    def get_widget(self):
        return self.wid

    def get_viewer(self,key):
        return self.viewers[key]

    def activate_viewer(self,key):
        viewer = self.viewers[key]
        self.wid.activate_viewer(viewer)
    
    def add_viewer(self,viewer:ViewerBase,name):
        self.viewers[name] = viewer
        self.wid.add_viewer(viewer,name)





class OperationManager():
    def __init__(self,res,ld,fld):
        self.res = res
        self.ld = ld
        self.fld = fld

        self.operations = {}
        self.current_operation = None
        self.viewer_keys = {}

        self.ope_dep = OperationDependence()
        self.wid = OperationTabWidget()
        self.viewer_manager = ViewerManager()


        self.wid.actionTriggered.connect(self.run)

    def get_widget(self):
        return self.wid
    def get_viewerwrapper_widget(self):
        return self.viewer_manager.get_widget()

    def run(self,key):
        self.current_operation = key
        ope = self.operations[key]
        viewer_key = self.viewer_keys[key]
        self.viewer_manager.activate_viewer(viewer_key)
        ope.run()
        if isinstance(ope,StaticOperation):
            self.finish_current()

    def load_operation_log(self,log):
        self.ope_dep.load(log)
        self.wid.disable_operations(self.ope_dep.get_noneligible())
        self.wid.enable_operations(self.ope_dep.get_eligible())
        self.wid.enable_operations(self.ope_dep.get_done())

    def add_operation(self,name,ope:Operation,viewer_key,deps=None):
        self.operations[name] =ope
        self.ope_dep.register(name,deps)
        self.wid.add_operation(name)
        self.viewer_keys[name]=viewer_key
        ope.finish_signal().connect(self.finish_current)
    
    def add_static_operation(self,name,ope:StaticOperation,deps=None):
        self.operations[name] =ope
        self.ope_dep.register(name,deps)
        self.wid.add_operation(name)
    
    def finish_current(self):
        self.operations[self.current_operation].post_finish()
        self.ope_dep.check(self.current_operation)
        self.current_operation = None
        self.wid.enable_operations(self.ope_dep.get_eligible())
        self.wid.enable_operations(self.ope_dep.get_done())


class MainWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.resize(1200,800)


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
        self.ope = OperationManager()
        self.wid = MainWidget()


        self.wid.save_signal().connect(self.save)
        self.wid.load_signal().connect(self.load)

    def get_widgets(self):
        return self.wid, self.ope.get_widget(), self.ope.get_viewerwrapper_widget()
    def save(self):
        self.res.log.update('operation_log',self.ope.export_operation_log())
        self.res.save()
    def load(self):
        self.res.load()
        status = self.res.log.by_key('operation_log')
        self.ope.load_operation_log(status)


def main():
    app = QApplication(sys.argv)
    impath = sys.argv[1]
    fallimpath = sys.argv[2]
    direc = sys.argv[3]
    if not os.path.exists(direc):
        os.mkdir(direc)

    m = MainControl(direc,impath,fallimpath)
    main_wid, ope_wid, vis_wid = m.get_window()
    main_wid.show()
    ope_wid.show()
    vis_wid.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()