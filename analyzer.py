import os
import sys
import json
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QPushButton
from json import JSONEncoder

from guitools import MainTabWindowBase
import operations as ope

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



class MainTabWindow(MainTabWindowBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.tab.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.resize(1200,800)
        # self.p = QPushButton('save')
        # self.metalayout.addWidget(self.p)
        # self.p2 = QPushButton('load')
        # self.metalayout.addWidget(self.p2)

    def addTab(self,wid,name,ix):
        if ix>=self.tab.count():
            self.tab.addTab(wid,name)
            self.tab.setCurrentWidget(wid)
            return
        self.tab.removeTab(ix)
        self.tab.insertTab(ix,wid,name)
        self.tab.setCurrentWidget(wid)


class DoneFlag():
    def __init__(self,logres:ResultsDict):
        self.ix = -1
        self.logres=logres
    def get(self):
        return self.ix
    def update(self,ix):
        if self.ix<ix:
            self.ix = ix
        self.logres.update('done',self.ix)

class MainTabControl():
    def __init__(self,impath,fallimpath,direc):
        self.window = MainTabWindow()
        self.direc = direc
        self.res = Results(direc)
        self.done = DoneFlag(self.res.log)
        self.res.log.update('done',self.done)
        # self.window.p.clicked.connect(self.res.save)
        # self.window.p2.clicked.connect(self.load)
        self.ld = ope.Loader(impath)
        self.fld = ope.Loader(fallimpath)

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
        self.fbc = ope.BackgroundCorrection(self.fld)
        self.fbc.calc()
        self.fmm = ope.MaskMaker(self.fld,self.fbc.get())
        self.fmm.calc()
        self.ftr = ope.TrackerMain(self.fmm.get())
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
        self.fci = ope.CircleFitControl(self.fld,df)
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
        self.gra = ope.GravityFitControl(self.fld,pos)
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
        self.bc = ope.BackgroundCorrection(self.ld)
        self.bc.calc()
        self.mm = ope.MaskMaker(self.ld,self.bc.get())
        self.mm.calc()
        self.tra = ope.TrackerMain(self.mm.get())
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
        self.cir = ope.CircleFitControl(self.ld,df)
        self.window.addTab(self.cir.get_window(),'circle',4)
        self.cir.finish_signal().connect(self.cir_fin)
    def cir_fin(self):
        self.done.update(4)
        df = self.cir.get_df()
        self.res.oned.add_df(df)
        self.stick_pos()
    def stick_pos(self):
        self.sti = ope.StickPositionEditorControl(self.ld)
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
        stipos = ope.StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        self.stif = ope.StickFinderControl(self.ld,df,stipos)
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
        self.sav = ope.FitPlotControl(lpos,rpos,dposlist)
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
        stipos = ope.StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        self.ca = ope.ChainAssignControl(self.ld,lpos,rpos,dposlist,stipos)
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
        self.frc = ope.ForceCalcControl(grav,acc,phi,theta)
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
        self.tenop = ope.TensionOptimization(chain,force,phi,tl,tr,tl_e,tr_e)
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
        stipos = ope.StickPosition(framenum)
        stipos.loadchanges(stifra,stista)
        stiposarr = stipos.get_array()
        self.view = ope.ResultsViewerControl(self.direc,self.ld,df,chain,grav,dianum,stiposarr,tension)
        self.window.addTab(self.view.get_window(),'result',9)
        self.done.update(9)


def test():
    app = QApplication(sys.argv)
    impath = './test/td2.mov'
    fallimpath = './test/td_fall.mov'
    direc = './test/pro2'
    if not os.path.exists(direc):
        os.mkdir(direc)
    m = MainTabControl(impath,fallimpath,direc)
    m.get_window().show()
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
    main()