import os
from abc import ABC, abstractmethod

import numpy as np
import numpy.ma as ma
import pandas as pd
import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QLabel, QPushButton, QLineEdit, QWidget

from operation_base import Operation
from movieimporter import Loader

class BGLoaderConfig():
    def __init__(self):
        self.bg_color = [0,0,0]
        self.bg_subtraction_mode = 'split'

class BGLoader():
    def __init__(self,loader:Loader):
        self.c = BGLoaderConfig()

        self.ld = loader
        self.forward = [None for i in range(self.ld.getframenum())]
        self.backward = [None for i in range(self.ld.getframenum())]
        self.masked = [None for i in range(self.ld.getframenum())]

    def calc(self):
        '''note that those are weighted uint8 (gray scale) masks.'''
        backSub = cv2.createBackgroundSubtractorMOG2()
        for fp in range(0,self.ld.framenum):
            frame = self.ld.getframe(fp)
            self.forward[fp] = backSub.apply(frame).astype(np.uint8)

        backSub = cv2.createBackgroundSubtractorMOG2()
        for fp in range(self.ld.framenum-1,-1,-1):
            frame = self.ld.getframe(fp)
            self.backward[fp] = backSub.apply(frame).astype(np.uint8)

        for fp in range(self.ld.getframenum()):
            frame = self.ld.getframe(fp)
            fgmask = self._select_mask(fp)
            masked = np.zeros_like(frame)
            for channel in range(frame.shape[2]):
                maarr = ma.array(frame[:,:,channel],mask=np.logical_not(fgmask),fill_value=self.c.bg_color[channel])
                masked[:,:,channel] = ma.filled(maarr).astype(np.uint8)
            self.masked[fp] = masked
    
    def _select_mask(self,fpos):
        if self.c.bg_subtraction_mode =='split':
            if fpos>self.ld.getframenum()/2:
                return self.forward[fpos]
            else:
                return self.backward[fpos]

    def getframe(self,fpos):
        return self.masked[fpos]
    
    def get_fgmask(self):
        return self.forward, self.backward

    def hasframe(self,fpos):
        return self.ld.hasframe(fpos)
    
    def getframenum(self):
        return self.ld.getframenum()



class TrackingCalculation():
    def __init__(self,bgld:BGLoader):
        self.bgld = bgld
        self.framenum = self.bgld.getframenum()
        self.result_dict = {}
    
    def set_keys(self,keys):
        '''result =  {key: ndarray[frame,(x,y,w,h)],}'''
        for key in keys:
            if key in self.result_dict.keys():
                continue
            self.result_dict[key] = np.zeros((self.framenum,4),dtype=int)

    def track(self,key,spos,epos,initBB):
        if not key in self.result_dict.keys():
            raise KeyError('key [{key}] not found in result_dict')
        self.result_dict[key][spos:epos,:] = self._tracking(spos,epos,initBB)[spos:epos,:]

    def _tracking(self,spos,epos,initBB):
        box = np.zeros((self.framenum,4))
        frame = self.bgld.getframe(spos)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, initBB)
        box[spos,:] = [int(v) for v in initBB]
        for fp in range(spos+1,epos):
            # print(f'tracking {fp}')
            frame = self.bgld.getframe(fp)
            success, b = tracker.update(frame)
            if not success:
                break
            x, y, w, h = [int(v) for v in b]
            box[fp,:] = (x,y,w,h)
        return box

    def get_dict(self):
        return self.result_dict

    def get_df(self):
        basename = ['_x','_y','_w','_h']
        dfs = []
        for key,val in self.result_dict.items():
            name = [key + n for n in basename]
            dfs.append(pd.DataFrame(val,columns=name))
        outdf = pd.concat((dfs),axis=1)
        return outdf

    def get_singledict(self,fpos):
        out = {key:val[fpos,:] for key,val in self.result_dict.items()}
        return out




class TrackingOperationWidget(QWidget):
    TargetSelected = pyqtSignal(str)

    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.target_combo = QComboBox(self)
        self.target_button = QPushButton('change target')
        self.target_button.clicked.connect(self._select_target)
        self.current_target_label = QLabel('current target: null')

        self.roi_button = QPushButton('determine ROI')
        
        self.finish_button = QPushButton('finish')

        self.l0.addWidget(self.target_combo)
        self.l0.addWidget(self.target_button)
        self.l0.addWidget(self.current_target_label)
        self.l0.addWidget(self.roi_button)
        self.l0.addWidget(self.finish_button)
    
    def set_targets(self,keys):
        self.target_combo.clear()
        for key in keys:
            self.target_combo.addItem(key)
    
    def approved_target(self,key):
        self.target_combo.setCurrentIndex(self.target_combo.findText(key))
        self.current_target_label.setText(f'current target: {key}')
    
    def _select_target(self):
        tgt = self.target_combo.currentText()
        self.TargetSelected.emit(tgt)
    
    def get_target_signal(self):
        return self.TargetSelected
    
    def get_roi_signal(self):
        return self.roi_button.clicked
    
    def get_finish_signal(self):
        return self.finish_button.clicked




class TrackingOperationBase(Operation,ABC):
    def __init__(self,res,ld):
        '''use SingleViewer'''
        super().__init__(res,ld)
        self.bgl = BGLoader(self.ld)
        self.calc = TrackingCalculation(self.bgl)
        self.wid = TrackingOperationWidget()


    @abstractmethod
    def _default_keys(self):
        '''set possible targets'''
        self.targets = ['d0']
        self.current_key = 'd0'

    def run(self):
        self.targets = []
        self.current_key = ''
        self._default_keys()
        self.wid.set_targets(self.targets)
        self.calc.set_keys(self.targets)
        self.wid.approved_target(self.current_key)

        self.bgl.calc()
        self.wid.get_roi_signal().connect(self._roi_event)
        self.wid.get_target_signal().connect(self._target_event)

        self.viewer.change_fpos(0)

    def post_finish(self):
        self.wid.get_roi_signal().disconnect(self._roi_event)
        self.wid.get_target_signal().disconnect(self._target_event)

        result_df = self.calc.get_df()
        self._save(result_df)

    @abstractmethod
    def _save(self,result_df):
        '''save result_df to self.res'''
        pass

    def finish_signal(self):
        return self.wid.get_finish_signal()
    def get_widget(self):
        return self.wid
    def viewer_setting(self,viewerset):
        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':1})
        self.viewerset.deploy('single')
        self.viewer = self.viewerset.get_viewers()['single'][0]

        self.viewer.set_loader(self.bgl)
        drawing = self.viewer.get_drawing()
        drawing.vis_off()

        self.viewer.setting.enable_roi = True
        self.viewer.setting.show_roi_bgr = False
        self.viewer.apply_setting()

        drawing.set_fpos(self.ld.getframenum())

        self._viewer_add_rec()


    def _roi_event(self):
        rec = self.viewer.get_roi().get_rectangle()
        self.calc.track(self.current_key,self.viewer.get_fpos(),self.bgl.getframenum(),rec)
        self._viewer_add_rec()
        self.viewer.change_fpos(self.viewer.get_fpos())

    def _target_event(self,key):
        self.current_key = key
        self.wid.approved_target(key)
        self._change_color()

    def _viewer_add_rec(self):
        drawing = self.viewer.get_drawing()
        rec_dict = self.calc.get_dict()
        for key,val in rec_dict.items():
            drawing.set_rectangle(key,val)
        self._change_color()

    def _change_color(self):
        drawing = self.viewer.get_drawing()
        sty = drawing.getsty()
        for unitkey,val in sty.items():
            if unitkey[-4:]!='_rec':
                continue
            if unitkey.split('_')[0]==self.current_key:
                val['pen']['color']='g'
            else:
                val['pen']['color']='c'
        drawing.setsty(sty)

class FallTrackingOperation(TrackingOperationBase):
    def _default_keys(self):
        '''set possible targets'''
        self.targets = ['d0']
        self.current_key = 'd0'
    def _save(self,result_df):
        resunit = self.res.get_unit('fall_tracking')
        resunit.set(result_df)

class TrackingOperation(TrackingOperationBase):
    def _default_keys(self):
        '''set possible targets'''
        bres = self.res.get_unit('basics').get()
        ndia = bres['ndia']
        self.targets = ['l','r']
        for i in range(ndia):
            self.targets.append('d'+str(i))
        self.current_key = 'l'

    def _save(self,result_df):
        resunit = self.res.get_unit('tracking')
        resunit.set(result_df)