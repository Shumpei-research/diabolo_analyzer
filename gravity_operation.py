from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import QFormLayout, QVBoxLayout, QLabel, QPushButton, QLineEdit, QWidget

from operation_base import Operation
from visualize import ViewerSet





@dataclass
class GravityConfig():
    curve_fineness:float = 0.1


class GravityEstimation():
    def __init__(self):
        self.config = GravityConfig()

    def set(self,position:np.ndarray,pos_bool:np.ndarray):
        '''ndarray[frame,(x,y)]'''
        self.dpos = position
        self.bool = pos_bool

    def calc(self,first,last):
        self.first = first
        self.last = last
        dpos = self.dpos[first:last,:]
        dead = np.logical_not(self.bool[first:last])
        frames = np.arange(first,last,dtype=int)
        dposx = np.delete(dpos[:,0],dead)
        dposy = np.delete(dpos[:,1],dead)
        frames = np.delete(frames,dead)

        self.xcoef,self.xres = self._gravity_fit(frames, dposx)
        self.ycoef,self.yres = self._gravity_fit(frames, dposy)
        self.accx = 2*self.xcoef[0]
        self.accy = 2*self.ycoef[0]
        self.gnorm = (self.accx**2 + self.accy**2)**0.5

        resarray = np.stack((frames,self.xres,self.yres),axis=1)
        self.res_df = pd.DataFrame(resarray,columns = ['frame','x','y'])

        self.curve_frames = np.arange(first,last,self.config.curve_fineness)
        self.xfit = np.polyval(self.xcoef,self.curve_frames)
        self.yfit = np.polyval(self.ycoef,self.curve_frames)
        fitarray = np.stack((self.curve_frames,self.xfit,self.yfit),axis=1)
        self.fit_df = pd.DataFrame(fitarray,columns=['frame','x','y'])

        dataarray = np.stack((frames,dposx,dposy),axis=1)
        self.data_df = pd.DataFrame(dataarray,columns = ['frame','x','y'])
    
    def get_gvector(self):
        ''' returns {x, y, norm} of g. pixel/frame^2 '''
        return {'x':self.accx, 'y':self.accy, 'norm':self.gnorm}

    def get_fit(self):
        '''
        fit_df: quadratic curve from fitting
        res_df: residues from the fit
        data_df: original data for the fitting
        columns=['frame','x','y'] for both
        '''
        return self.fit_df, self.res_df, self.data_df
    
    def get_frames(self):
        return self.first, self.last

    def _gravity_fit(self,x,y):
        coef = np.polyfit(x,y,deg=2)
        fit = np.polyval(coef,x)
        res = y - fit
        return coef,res






class GravityOperationWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.fl = QFormLayout()
        self.l0.addLayout(self.fl)
        
        self.line1 = QLineEdit()
        self.line2 = QLineEdit()
        self.fl.addRow('beginning frame',self.line1)
        self.fl.addRow('end frame',self.line2)

        self.fit_button = QPushButton('fit')
        self.res_label = QLabel('')
        self.l0.addWidget(self.fit_button)
        self.l0.addWidget(self.res_label)

        self.finish_button = QPushButton('finish')
        self.l0.addWidget(self.finish_button)
    
    def finish_signal(self):
        return self.finish_button.clicked
    def fit_signal(self):
        return self.fit_button.clicked
    
    def get_frames(self):
        begin = int(self.line1.text())
        end = int(self.line2.text())
        return begin,end
    def show_result(self,txt:str):
        self.res_label.setText(txt)






class GravityOperation(Operation):
    def __init__(self,res,ld):
        super().__init__(res,ld)
        self.wid = GravityOperationWidget()
        self.calc = GravityEstimation()

    def run(self):
        '''perform calculation/interactive operation'''
        unit = self.res.get_unit('fall_circle')
        circledf = unit.get()
        self.pos = circledf.loc[:,('d0_x','d0_y')].values
        self.pos_bool = circledf['d0_r'].values != 0
        self.calc.set(self.pos,self.pos_bool)

        self.wid.fit_signal().connect(self._fit_event)
        self.viewer.change_fpos(0)

    def post_finish(self):
        '''will be called after finishing operation to take out data.'''
        self.wid.fit_signal().disconnect(self._fit_event)

        unit = self.res.get_unit('gravity')
        first,last = self.calc.get_frames()
        gvector = self.calc.get_gvector()
        res_dict = {'frame_range':[first,last],'gvector':gvector}
        unit.update(res_dict)

    def finish_signal(self):
        return self.wid.finish_signal()

    def get_widget(self):
        return self.wid
        
    def viewer_setting(self,viewerset:ViewerSet):
        '''set viewer'''
        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':1,'splot':2})
        self.viewerset.deploy('grid',rows={'single':1,'splot':2},
            cols={'single':1,'splot':1},order={'single':'cols','splot':'rows'})
        self.viewer = self.viewerset.get_viewers()['single'][0]
        self.plots = self.viewerset.get_viewers()['splot']
        self.plis = [p.get_pli() for p in self.plots]

        self.viewer.set_loader(self.ld)

        drawing = self.viewer.get_drawing()
        drawing.vis_off()

        self.viewer.setting.enable_roi = False
        self.viewer.setting.show_roi_bgr = False
        self.viewer.apply_setting()

        for pli in self.plis:
            pli.setLabel('bottom','frame')
        self.plis[0].setLabel('left','x (pixel)')
        self.plis[1].setLabel('left','y (pixel)')

        self.p0d = self.plis[0].plot([],[],symbol='o',
            symbolPen={'color':'w','width':1},symbolBrush=None,symbolSize=5,pen=None)
        self.p0f = self.plis[0].plot([],[],symbol=None,
            pen={'color':'w','width':1})
        self.p1d = self.plis[1].plot([],[],symbol='o',
            symbolPen={'color':'w','width':1},symbolBrush=None,symbolSize=5,pen=None)
        self.p1f = self.plis[1].plot([],[],symbol=None,
            pen={'color':'w','width':1})
    
    def _fit_event(self):
        begin,end = self.wid.get_frames()
        self.calc.calc(begin,end)
        self._show_gvector()
        self._show_fit()
    
    def _show_gvector(self):
        gvector = self.calc.get_gvector()
        self.wid.show_result(json.dumps(gvector))

    def _show_fit(self):
        fit, residue, data = self.calc.get_fit()
        self.p0d.setData(data['frame'].values,data['x'].values)
        self.p0f.setData(fit['frame'].values,fit['x'].values)
        self.p1d.setData(data['frame'].values,data['y'].values)
        self.p1f.setData(fit['frame'].values,fit['y'].values)


