import pyqtgraph as pg
import numpy as np
import os
import pyqtgraph.exporters
import pandas as pd
import json
from PyQt5.QtCore import Qt

from guitools import ViewerBase,ViewControlBase
from visualize import NewDrawing
from utilities import StickPosition
from movieimporter import Loader

class ResultsViewerWidget(ViewerBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.drawing = NewDrawing(self.pli)
        self.pli.showAxis('left',False)
        self.pli.showAxis('bottom',False)
    def get_drawing(self):
        return self.drawing


class ViewerSetting():
    '''Apply setting to viewer widget using results object'''
    def __init__(self,wid,res):
        self.wid = wid
        self.drawing = self.wid.get_drawing()
        self.res = res


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
        posdict = {}
        for key in objectkeys:
            name = [key+'_savgol_x',key+'_savgol_y']
            posdict[key]=df[name].values
        self.window.drawing.set_positions(posdict)
        self.window.drawing.set_string_tension(self._chain()[0],posdict,tension)
        lacc = self.df[['l_ax','l_ay']].values
        racc = self.df[['r_ax','r_ay']].values
        dforcelist = []
        for i in range(self.dianum):
            key = 'd'+str(i)
            name = [key+'_optforce_x',key+'_optforce_y']
            dforcelist.append(self.df[name].values)
        lflyframes,rflyframes = self._flyframes()
        forcedict = {'l':(lacc-self.grav[0:2])/self.grav[2],'r':(racc-self.grav[0:2])/self.grav[2]}
        for i in range(len(dforcelist)):
            forcedict['d'+str(i)] = dforcelist[i]
        self.window.drawing.set_force(self.grav,forcedict,posdict,lflyframes,rflyframes)

        torquedict = {}
        for i in range(self.dianum):
            key = 'd'+str(i)
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