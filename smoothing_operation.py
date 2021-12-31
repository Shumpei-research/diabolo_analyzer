import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.signal
import scipy.interpolate
import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QFormLayout, QTabWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QWidget

from operation_base import Operation
from movieimporter import Loader
from utilities import CenterToBox, ConfigEditor, StickPosition, CtoBsingle
from visualize import ViewerSet


@dataclass
class SmoothingConfig():
    savgol_window:int = 31
    savgol_poly:int = 6
    savgol_mode:str = 'nearest'
    savgol_pre_window:int = 11
    savgol_pre_poly:int = 3
    savgol_pre_mode:str = 'nearest'
    savgol_pre_sigma:int = 2

@dataclass
class SmoothingStickConfig():
    savgol_window:int = 25
    savgol_poly:int = 4
    savgol_mode:str = 'nearest'
    savgol_pre_window:int = 15
    savgol_pre_poly:int = 3
    savgol_pre_mode:str = 'nearest'
    savgol_pre_sigma:int = 3



class SmoothingSavgol():
    def __init__(self):
        pass
    def set_config(self,config_dict):
        self.c = SmoothingConfig(**config_dict)
    def get_config(self):
        return asdict(self.c)

    def set(self,config,pos,posbool):
        self.c=config
        self.pos = pos
        self.frames = np.arange(self.pos.shape[0],dtype=int)
        self.posbool = posbool

        fnum = self.pos.shape[0]
        self.out_bool = np.zeros((fnum,2),dtype=bool)
        self.robust_pos = np.zeros((fnum,2),dtype=float) # pos in robust filtering
        self.residue = np.zeros((fnum,2),dtype=float)
        self.border = [0.0,0.0] # border for robust filtering
        self.robust_filtered_pos = np.zeros((fnum,2),dtype=float) # pos after robust filtering
        self.savgol = np.zeros((fnum,2,3),dtype=float) # ax3: (x,v,a)
    
    def get_final(self):
        '''ndarray[frame,(x,y),(x,v,a)]'''
        return self.savgol
    def get_intermediate(self):
        out = {
            'fit':self.robust_pos,
            'outlier':self.out_bool,
            'residue':self.residue,
            'border':self.border,
            'corrected':self.robust_filtered_pos
        }
        return out
        

    def calc(self):
        self.f = self.frames[self.posbool]
        for i in range(2):
            x = self.pos[:,i][self.posbool]
            res = self._rob_savgol(x)
            self.robust_filtered_pos[:,i] = res[0]
            self.residue[:,i] = res[1]
            self.border[i] = res[2]
            self.out_bool[:,i] = res[3]
            self.robust_pos[:,i] = res[4]
        
        res = self._interp_savgol(self.robust_filtered_pos[:,0],self.robust_filtered_pos[:,1])
        self.savgol[:,0,:] = np.stack((res[0],res[2],res[4]),axis=1)
        self.savgol[:,1,:] = np.stack((res[1],res[3],res[5]),axis=1)

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

        f = np.arange(x.shape[0])
        newf = f[~out]
        interpolated = scipy.interpolate.pchip_interpolate(newf,x[~out],f)
        new[out]=interpolated[out]

        return new,res,border,out,smooth






class SmoothingWrapper():
    def __init__(self):
        pass
    def get_final(self):
        out = {}
        for i,d in enumerate(self.dia_calc):
            out['d'+str(i)] = d.get_final()
        out['l'] = self.stick_calc[0].get_final()
        out['r'] = self.stick_calc[1].get_final()
        return out
    def get_intermediate(self):
        out = {}
        for i,d in enumerate(self.dia_calc):
            out['d'+str(i)] = d.get_intermediate()
        out['l'] = self.stick_calc[0].get_intermediate()
        out['r'] = self.stick_calc[1].get_intermediate()
        return out

    def set_dia(self,pos_list,posbool_list):
        self.ndia = len(pos_list)
        self.dia_calc = [SmoothingSavgol() for i in range(self.ndia)]
        for i in range(self.ndia):
            c = SmoothingConfig()
            self.dia_calc[i].set(c,pos_list[i],posbool_list[i])

    def set_stick(self,pos_list,posbool_list):
        self.stick_calc = [SmoothingSavgol() for i in range(2)]
        for i in range(2):
            c = SmoothingStickConfig()
            self.stick_calc[i].set(c,pos_list[i],posbool_list[i])
    
    def main(self):
        for c in self.dia_calc:
            c.calc()
        for c in self.stick_calc:
            c.calc()
    
    def get(self):
        pass
    
    def set_config_dia_sub(self,config_dict,index):
        self.dia_calc[index].set_config(config_dict)
    def set_config_stick_sub(self,config_dict,index):
        self.stick_calc[index].set_config(config_dict)
    def set_config_dia(self,config_dict):
        for i in range(self.ndia):
            self.set_config_dia_sub(copy.deepcopy(config_dict),i)
    def set_config_stick(self,config_dict):
        for i in range(2):
            self.set_config_stick_sub(copy.deepcopy(config_dict),i)
    def get_config_dia_sub(self,index):
        return self.dia_calc[index].get_config()
    def get_config_stick_sub(self,index):
        return self.stick_calc[index].get_config()
    def get_config_dia(self):
        return self.get_config_dia_sub(0)
    def get_config_stick(self):
        return self.get_config_stick_sub(0)





class SmoothingWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.tab = QTabWidget()
        self.l0.addWidget(self.tab)

        self.editor = []
        for i in range(2):
            self.editor.append(ConfigEditor(self))
        self.tab.addTab(self.editor[0],'diabolo')
        self.tab.addTab(self.editor[1],'stick')

        self.finish_button = QPushButton('finish')
        self.l0.addWidget(self.finish_button)

    def set_config(self,config:dict,index):
        self.editor[index].setdict(config)

    def update_signal(self,index):
        '''pyqtSignal(dict)'''
        return self.editor[index].UpdatedConfig
    
    def finish_signal(self):
        return self.finish_button.clicked




class SmoothingPlotter():
    '''utility class for plotting smoothing processes'''
    def __init__(self,plots,posbool):
        self.pli = plots
        self.bool = posbool

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
        f=f[self.bool]
        x=x[self.bool]
        xn=xn[self.bool]
        vx=vx[self.bool]
        ax=ax[self.bool]

        self.p0.setData(f,ax)
        self.p1.setData(f,vx)
        self.p2.setData(f,xn)
        self.p2dot.setData(f,x)
        self.p3dot.setData(f,xn-x)
    def draw_pre(self,f,x,pre,res,border,out):
        f=f[self.bool]
        x=x[self.bool]
        pre=pre[self.bool]
        res=res[self.bool]
        out=out[self.bool]

        self.p4.setData(f,pre)
        self.p4dot0.setData(f[out],x[out])
        self.p4dot1.setData(f[~out],x[~out])
        self.p5dot0.setData(f[~out],res[~out])
        self.p5dot1.setData(f[out],res[out])
        self.p5line0.setValue(border)
        self.p5line1.setValue(-border)




class SmoothingOperation(Operation):
    def __init__(self,res,ld):
        super().__init__(res,ld)
        self.wid = SmoothingWidget()
        self.calc = SmoothingWrapper()


    def run(self):
        self.calc.set_dia(self.dposlist,self.dboollist)
        self.calc.set_stick(self.sposlist,self.sboollist)

        self.calc.main()

        self.wid.set_config(self.calc.get_config_dia(),0)
        self.wid.set_config(self.calc.get_config_stick(),1)

        self.wid.update_signal(0).connect(lambda x: self._update_config(0,x))
        self.wid.update_signal(1).connect(lambda x: self._update_config(1,x))

        self._set_view()

    def post_finish(self):
        self.wid.update_signal(0).disconnect()
        self.wid.update_signal(1).disconnect()

        res_dict = self.calc.get_final()
        dfs = []
        basename = ['_x','_y','_vx','_vy','_ax','_ay']
        for key,val in res_dict.items():
            name = [key+n for n in basename]
            arr = np.concatenate((val[:,:,0], val[:,:,1], val[:,:,2]),axis=1)
            dfs.append(pd.DataFrame(arr,columns=name))
        res_df = pd.concat(dfs,axis='columns')

        self.res.get_unit('smoothened').add_df(res_df)

    def finish_signal(self):
        return self.wid.finish_signal()

    def get_widget(self):
        return self.wid

    def viewer_setting(self,viewerset:ViewerSet):

        self.ndia = self.res.get_unit('basics').get()['ndia']
        circleunit = self.res.get_unit('circle')
        circle_df = circleunit.get()
        basename = ['_x','_y']
        self.dposlist = [None for i in range(self.ndia)]
        self.dboollist = [None for i in range(self.ndia)]
        for d in range(self.ndia):
            name =['d'+str(d)+n for n in basename]
            self.dposlist[d] = circle_df[name].values
            self.dboollist[d] = np.any(self.dposlist[d]!=0,axis=1)
        
        sunit = self.res.get_unit('stick')
        spos_df = sunit.get()
        lpos = spos_df[['l_x','l_y']].values
        lbool = np.any(lpos!=0,axis=1)
        rpos = spos_df[['r_x','r_y']].values
        rbool = np.any(rpos!=0,axis=1)
        self.sposlist = [lpos,rpos]
        self.sboollist = [lbool,rbool]


        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':1,'splot':12*(self.ndia+2)})
        tn = ['left','right'] + ['d'+str(i) for i in range(self.ndia)]
        self.viewerset.deploy('tab',rows={'single':1,'splot':6},
            cols={'single':1,'splot':2},tabnames = {'single':['movie'],'splot':tn},
            order='rows')
        vs = self.viewerset.get_viewers()
        self.viewer = vs['single'][0]
        self.l_pli = [p.get_pli() for p in vs['splot'][0:12]]
        self.r_pli = [p.get_pli() for p in vs['splot'][12:24]]
        self.d_pli = [None for i in range(self.ndia)]
        for d in range(self.ndia):
            self.d_pli[d] = [p.get_pli() for p in vs['splot'][24+12*d:36+12*d]]

        self.viewer.set_loader(self.ld)
        self.viewer.setting.enable_roi = False
        self.viewer.setting.show_roi_bgr = False
        self.viewer.apply_setting()
        drawing = self.viewer.get_drawing()
        drawing.vis_off()

        self.l_x_plt = SmoothingPlotter(self.l_pli[0:6],self.sboollist[0])
        self.l_y_plt = SmoothingPlotter(self.l_pli[6:12],self.sboollist[0])
        self.r_x_plt = SmoothingPlotter(self.r_pli[0:6],self.sboollist[1])
        self.r_y_plt = SmoothingPlotter(self.r_pli[6:12],self.sboollist[1])
        self.d_x_plt = [None for i in range(self.ndia)]
        self.d_y_plt = [None for i in range(self.ndia)]
        for i in range(self.ndia):
            self.d_x_plt[i] = SmoothingPlotter(self.d_pli[i][0:6],self.dboollist[i])
            self.d_y_plt[i] = SmoothingPlotter(self.d_pli[i][6:12],self.dboollist[i])

    
    def _update_config(self,index,config_dict):
        '''index=0: diabolo, index=1: stick'''
        if index==0:
            self.calc.set_config_dia(config_dict)
        elif index ==1:
            self.calc.set_config_stick(config_dict)
        self.calc.main()
        self._set_view()
    
    def _set_view(self):
        frames = np.arange(self.ld.getframenum(),dtype=int)
        interm = self.calc.get_intermediate()
        final = self.calc.get_final()

        intermsub = interm['l']
        out = final['l']

        self.l_x_plt.draw_pre(frames,self.sposlist[0][:,0],intermsub['fit'][:,0],intermsub['residue'][:,0],
            intermsub['border'][0],intermsub['outlier'][:,0])
        self.l_y_plt.draw_pre(frames,self.sposlist[0][:,1],intermsub['fit'][:,1],intermsub['residue'][:,1],
            intermsub['border'][1],intermsub['outlier'][:,1])

        self.l_x_plt.draw_diff(frames,intermsub['corrected'][:,0],out[:,0,0],
            out[:,0,1],out[:,0,2])
        self.l_y_plt.draw_diff(frames,intermsub['corrected'][:,1],out[:,1,0],
            out[:,1,1],out[:,1,2])

        intermsub = interm['r']
        out = final['r']

        self.r_x_plt.draw_pre(frames,self.sposlist[1][:,0],intermsub['fit'][:,0],intermsub['residue'][:,0],
            intermsub['border'][0],intermsub['outlier'][:,0])
        self.r_y_plt.draw_pre(frames,self.sposlist[1][:,1],intermsub['fit'][:,1],intermsub['residue'][:,1],
            intermsub['border'][1],intermsub['outlier'][:,1])
        self.r_x_plt.draw_diff(frames,intermsub['corrected'][:,0],out[:,0,0],
            out[:,0,1],out[:,0,2])
        self.r_y_plt.draw_diff(frames,intermsub['corrected'][:,1],out[:,1,0],
            out[:,1,1],out[:,1,2])
        
        for d in range(self.ndia):
            intermsub = interm['d'+str(d)]
            out = final['d'+str(d)]

            self.d_x_plt[d].draw_pre(frames,self.dposlist[d][:,0],intermsub['fit'][:,0],intermsub['residue'][:,0],
                intermsub['border'][0],intermsub['outlier'][:,0])
            self.d_y_plt[d].draw_pre(frames,self.dposlist[d][:,1],intermsub['fit'][:,1],intermsub['residue'][:,1],
                intermsub['border'][1],intermsub['outlier'][:,1])

            self.d_x_plt[d].draw_diff(frames,intermsub['corrected'][:,0],out[:,0,0],
                out[:,0,1],out[:,0,2])
            self.d_y_plt[d].draw_diff(frames,intermsub['corrected'][:,1],out[:,1,0],
                out[:,1,1],out[:,1,2])
        
        self.viewer.change_fpos(0)
    