from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QLabel, QPushButton, QLineEdit, QWidget

from operation_base import Operation
from movieimporter import Loader
from utilities import CenterToBox, ConfigEditor
from visualize import ViewerSet







@dataclass
class CircleConfig:
    box_size: int = 40
    grad_p1: float = 1.0
    grad_p2: float = 1.0
    hough_p1: float = 300.0
    hough_p2: float = 1.0
    minrad: int = 5
    maxrad: int = 10
    mindist:float  = 7.5


class CircleCalculation():
    '''[center_x,center_y,radius]'''
    def __init__(self,ld:Loader,dia_rec_list):
        self.c = CircleConfig()
        self.ld = ld
        self.dia_rec_list = dia_rec_list
        self.dia_rec_bool_list = [np.any(rec,axis=1) for rec in self.dia_rec_list]
        self.ndia = len(self.dia_rec_list)
        self.dia_circle_list = [np.zeros((self.ld.getframenum(),3)) for i in range(self.ndia)]
        self.boundbox = [np.zeros((self.ld.getframenum(),4)) for i in range(self.ndia)]

    def get_list(self):
        return self.dia_circle_list
    def get_bb(self):
        return self.boundbox
    def get_df(self):
        subdf = []
        basename = ['_x','_y','_r']
        for i in range(self.ndia):
            name = ['d'+str(i)+n for n in basename]
            subdf.append(pd.DataFrame(self.dia_circle_list[i],columns=name))
        out = pd.concat(subdf,axis=1)
        return out

    def get_config(self):
        return asdict(self.c)
    def set_config(self,config:dict):
        self.c = CircleConfig(**config)

    def calc(self):
        frame_shape = self.ld.getframe(0).shape
        for d in range(self.ndia):
            ctb = CenterToBox(self.c.box_size,self.c.box_size,
                frame_shape,self.dia_rec_list[d])
            ctb.calc()
            self.boundbox[d] = ctb.get_box()
            for fp in range(0,self.ld.getframenum()):
                if not self.dia_rec_bool_list[d][fp]:
                    continue
                box = self.boundbox[d][fp,:]
                frame = self.ld.getframe(fp)
                circles = self._circle(frame,box)
                if circles is None:
                    continue
                if circles.ndim!=2:
                    continue
                i=0
                cent = (circles[i,0],circles[i,1])
                rad = circles[i,2]
                if rad==0:
                    continue
                self.dia_circle_list[d][fp,0:2]=cent
                self.dia_circle_list[d][fp,2] = rad

    def _circle(self,frame,box):
        (x,y,w,h) = box
        roi = frame[y:y+h,x:x+w,:]
        gr = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gr,cv2.HOUGH_GRADIENT,
            self.c.grad_p1, self.c.grad_p2, self.c.mindist, self.c.hough_p1, self.c.hough_p2,
            minRadius = self.c.minrad, maxRadius = self.c.maxrad)
        if circles.ndim==3:
            circles = circles[0]
        elif circles.ndim==2:
            return None
        circles[:,0] = circles[:,0]+x
        circles[:,1] = circles[:,1]+y
        return circles




class CircleOperationWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.editor = ConfigEditor(self)
        self.l0.addWidget(self.editor)

        self.finish_button = QPushButton('finish')
        self.l0.addWidget(self.finish_button)
    
    def set_config(self,config:dict):
        self.editor.setdict(config)

    def update_signal(self):
        '''pyqtSignal(dict)'''
        return self.editor.UpdatedConfig
    
    def finish_signal(self):
        return self.finish_button.clicked



class CircleOperationBase(Operation,ABC):
    def __init__(self,res,ld):
        '''use MultiViewer'''
        super().__init__(res,ld)
        self.wid = CircleOperationWidget()

    @abstractmethod
    def _box_list(self):
        pass

    def run(self):
        self.wid.update_signal().connect(self._update_event)

        dia_rec_list = self._box_list()
        self.ndia = len(dia_rec_list)
        self.dia_rec_bool_list = [np.any(rec,axis=1) for rec in dia_rec_list]
        self.calc = CircleCalculation(self.ld,dia_rec_list)
        self.wid.set_config(self.calc.get_config())

        self.calc.calc()
        self._set_crop()
        self._viewer_add_circle()

        self.single_viewers[0].change_fpos(0)

    def post_finish(self):
        self.wid.update_signal().disconnect(self._update_event)

        result_df = self.calc.get_df()
        self._save(result_df)

    @abstractmethod
    def _save(self,result_df):
        '''save result_df to self.res'''
        pass

    def finish_signal(self):
        return self.wid.finish_signal()

    def get_widget(self):
        return self.wid

    def viewer_setting(self,viewerset:ViewerSet):
        dia_rec_list = self._box_list()
        self.ndia = len(dia_rec_list)

        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':self.ndia})
        self.viewerset.deploy('grid',rows={'single':self.ndia},
            cols={'single':1},order='cols')
        self.single_viewers = self.viewerset.get_viewers()['single']

        for i in range(1,len(self.single_viewers)):
            self.single_viewers[0].link_frame(self.single_viewers[i])

        for viewer in self.single_viewers:
            viewer.set_loader(self.ld)
            viewer.setting.enable_roi = False
            viewer.setting.show_roi_bgr = False
            viewer.apply_setting()

            drawing = viewer.get_drawing()
            drawing.vis_off()

    
    def _set_crop(self):
        bblist = self.calc.get_bb()
        for d,viewer in enumerate(self.single_viewers):
            viewer.set_cropbox(bblist[d],self.dia_rec_bool_list[d])

    def _viewer_add_circle(self):
        circle_list = self.calc.get_list()
        for i,arr in enumerate(circle_list):
            drawing = self.single_viewers[i].get_drawing()
            key = 'd'+str(i)
            drawing.set_circle(key,arr[:,0:2],arr[:,2])
    
    def _update_event(self,config_dict):
        self.calc.set_config(config_dict) # rld.set_config() is included.
        self.calc.calc()
        self._set_crop()
        self._viewer_add_circle()

class CircleOperation(CircleOperationBase):
    def _box_list(self):
        ndia = self.res.get_unit('basics').get()['ndia']
        boxresunit = self.res.get_unit('tracking')
        box_df = boxresunit.get()
        basename = ['_x','_y','_w','_h']
        rec_list = []
        for i in range(ndia):
            name = ['d'+str(i) + n for n in basename]
            arr = box_df[name].values
            rec_list.append(arr)
        return rec_list
    def _save(self,result_df):
        unit = self.res.get_unit('circle')
        unit.add_df(result_df)

class FallCircleOperation(CircleOperationBase):
    def _box_list(self):
        boxresunit = self.res.get_unit('fall_tracking')
        box_df = boxresunit.get()
        basename = ['_x','_y','_w','_h']
        name = ['d0'+n for n in basename]
        rec_list = [box_df[name].values]
        return rec_list
    def _save(self,result_df):
        unit = self.res.get_unit('fall_circle')
        unit.add_df(result_df)
