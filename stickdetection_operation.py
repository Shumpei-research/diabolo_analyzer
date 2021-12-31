from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QFormLayout, QVBoxLayout, QLabel, QPushButton, QLineEdit, QWidget

from operation_base import Operation
from movieimporter import Loader
from utilities import CenterToBox, ConfigEditor, StickPosition, CtoBsingle
from visualize import ViewerSet






class HsvLoader():
    def __init__(self):
        pass
    def set_loader(self,ld):
        self.ld = ld
    def calc(self):
        self.hsv = [None for i in range(self.ld.getframenum())]
        for fp in range(self.ld.getframenum()):
            im = self.ld.getframe(fp)
            self.hsv[fp] = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

    def hasframe(self,fpos):
        return self.ld.hasframe(fpos)
    def getframe(self,fpos):
        return self.hsv[fpos]
    def getframenum(self):
        return self.ld.getframenum()




@dataclass
class StickDetectionConfig():
    hcen: int = 170
    hwid: int = 10
    smin: int = 70
    smax: int = 255
    vmin: int = 70
    vmax: int = 255
    sticklen: int=100
    dilation: int=5
    minsize: int=120
    searchlen: int=100
    dilation_fly: int=7
    minsize_fly: int=100


class StickDetection():
    '''
    where sticks belong (left/right/flying) will be provided
    detect regions
    clean up regions
    if left and right:
        see if each region belongs to left or right
    if flying:
        'flying' region will be tracked
        'flying' regions are ignored in the following steps
    if left(/right) and flying:
        operate on remaining regions
    '''
    def __init__(self):
        self.config = StickDetectionConfig()

    def set(self,hsv_loader,lbox:np.ndarray,rbox:np.ndarray,
            lbool:np.ndarray,rbool:np.ndarray,stickpos:StickPosition):
        '''lbox/rbox: ndarray[frame,(x,y,w,h)]'''
        self.hsv_loader = hsv_loader
        self.frame_shape = self.hsv_loader.getframe(0).shape[0:2]

        self.lbox = lbox
        self.lpos = np.stack((lbox[:,0]+lbox[:,2]/2, lbox[:,1]+lbox[:,3]/2),axis=1)
        self.lbool = lbool

        self.rbox = rbox
        self.rpos = np.stack((rbox[:,0]+rbox[:,2]/2, rbox[:,1]+rbox[:,3]/2),axis=1)
        self.rbool = rbool

        self.stickpos = stickpos

        fnum = self.hsv_loader.getframenum()
        self.filtered_region = [np.array([],dtype=bool) for i in range(fnum)]
        self.cleaned_region = [np.array([],dtype=bool) for i in range(fnum)]
        self.lhand_bb = np.zeros((fnum,4),dtype=int)
        self.rhand_bb = np.zeros((fnum,4),dtype=int)
        self.overlap_bool = np.zeros(fnum,dtype=bool)
        self.lstick_reg = [np.array([],dtype=bool) for i in range(fnum)]
        self.rstick_reg = [np.array([],dtype=bool) for i in range(fnum)]
        self.lstick_pos = np.zeros((fnum,2))
        self.rstick_pos = np.zeros((fnum,2))

        self.lsearch_bb = np.zeros((fnum,4),dtype=int)
        self.rsearch_bb = np.zeros((fnum,4),dtype=int)

    def set_config(self,config_dict:dict):
        self.config = StickDetectionConfig(**config_dict)
    def get_config(self):
        return asdict(self.config)
    
    def get_pos_array(self):
        '''returns stick pos'''
        return self.lstick_pos, self.rstick_pos
    
    def get_pos_df(self):
        '''returns stick pos'''
        name = ['l_x','l_y','r_x','r_y']
        arr = np.concatenate((self.lstick_pos,self.rstick_pos),axis=1)
        df = pd.DataFrame(arr,columns=name)
        return df
    
    def get_reg(self):
        '''returns stick regions. (left stick, right stick)'''
        return self.lstick_reg, self.rstick_reg
    
    def get_bb(self):
        '''returns bb used for searching'''
        return self.lsearch_bb, self.rsearch_bb

    def main(self):
        self._colorfilter()
        self._check_overlap()
        self._makebb()
        self._distancemap()
        for s,e,l,r in self.stickpos.get_iter():
            state = [l,r]
            if np.any(state==[0,0]):
                # anything absent
                continue
            elif np.all(state==[1,2]):
                # held
                self._both(s,e)
            elif np.all(state==[2,1]):
                # reverse held
                self._both(s,e,reverse=True)
            elif state[0]==3 and state[1]==3:
                # both flying
                self._flying('l',s,e)
                self._flying('r',s,e)
            elif state[0]==3:
                # left flying right held by either rhand or lhand
                self._flying('l',s,e)
                if state[1]==2:
                    self._onehand('r','r',s,e)
                elif state[1]==1:
                    self._onehand('r','l',s,e)
            elif state[1]==3:
                # right flying left held by either rhand or lhand
                self._flying('r',s,e)
                if state[0]==2:
                    self._onehand('l','r',s,e)
                elif state[0]==1:
                    self._onehand('l','l',s,e)

    def _colorfilter(self):
        for fpos in range(self.hsv_loader.getframenum()):
            ch = self.hsv_loader.getframe(fpos)

            hrange = np.arange(self.config.hcen-self.config.hwid ,
                self.config.hcen+self.config.hwid+1,dtype=int)
            hrange = np.where(hrange<0,hrange+180,hrange)
            hrange = np.where(hrange>179,hrange-180,hrange)
            hreg = np.isin(ch[:,:,0],hrange)

            srange = [self.config.smin,self.config.smax]
            vrange = [self.config.vmin,self.config.vmax]
            sreg = np.logical_and(srange[0]<=ch[:,:,1],srange[1]>=ch[:,:,1])
            vreg = np.logical_and(vrange[0]<=ch[:,:,2],vrange[1]>=ch[:,:,2])
            
            region = np.all(np.stack((hreg,vreg,sreg),axis=0),axis=0)
            self.filtered_region[fpos] = region

            # cleanup
            region_cv = region.astype(np.uint8)
            kernel = np.ones((self.config.dilation,self.config.dilation),np.uint8)
            dilated = cv2.dilate(region_cv,kernel,iterations=1)
            
            connectivity = np.ones((3,3))
            lab, ln = ndi.label(dilated,structure=connectivity)
            un, counts = np.unique(lab,return_counts=True)
            ok_label = un[counts>self.config.minsize]
            ok_label = ok_label[ok_label!=0]
            cleaned_region = np.isin(lab,ok_label)
            self.cleaned_region[fpos]=cleaned_region

    def _makebb(self):
        ctb = CenterToBox(2*self.config.sticklen,2*self.config.sticklen,
            self.frame_shape,center=self.lpos)
        ctb.calc()
        self.lhand_bb = ctb.get_box()
        ctb = CenterToBox(2*self.config.sticklen,2*self.config.sticklen,
            self.frame_shape,center=self.rpos)
        ctb.calc()
        self.rhand_bb = ctb.get_box()
    
    def _check_overlap(self):
        diff = np.absolute(self.lpos - self.rpos)
        self.overlap_bool = np.all((diff<=2*self.config.sticklen),axis=1)

    def _distancemap(self):
        l = self.config.sticklen
        ri,ci = np.ogrid[-l:l+1, -l:l+1]
        map = ri*ri + ci*ci
        self.distance_map = map

    def _both(self,start,end,reverse=False):
        if reverse:
            self.lsearch_bb[start:end,:] = self.rhand_bb[start:end,:]
            self.rsearch_bb[start:end,:] = self.lhand_bb[start:end,:]
        else:
            self.lsearch_bb[start:end,:] = self.lhand_bb[start:end,:]
            self.rsearch_bb[start:end,:] = self.rhand_bb[start:end,:]

        for fp in range(start,end):
            if not (self.lbool[fp] and self.rbool[fp]):
                continue
            if self.overlap_bool[fp]:
                mreg = self._merge_box(fp)
                focused_region = np.logical_and(self.cleaned_region[fp],mreg)

                labeled,lnum = ndi.label(focused_region)
                regcent = ndi.center_of_mass(focused_region,labeled,np.arange(1,lnum+1))
                regcent = np.array([[r[1],r[0]] for r in regcent]) # x,y
                ldisp = regcent - np.expand_dims(self.lpos[fp,:],axis=0)
                rdisp = regcent - np.expand_dims(self.rpos[fp,:],axis=0)
                ldistance = np.linalg.norm(ldisp,axis=1)
                rdistance = np.linalg.norm(rdisp,axis=1)
                isleft = ldistance<=rdistance
                l_label = np.arange(1,lnum+1)[isleft]
                r_label = np.arange(1,lnum+1)[~isleft]
                l_region = np.isin(labeled,l_label)
                r_region = np.isin(labeled,r_label)
            else:
                l_region = np.logical_and(self.cleaned_region[fp],self._get_box(fp,'l'))
                r_region = np.logical_and(self.cleaned_region[fp],self._get_box(fp,'r'))
            
            if reverse:
                self.lstick_reg[fp] = r_region
                self.rstick_reg[fp] = l_region
                self._find_sticktip(fp,stick='l',hand='r')
                self._find_sticktip(fp,stick='r',hand='l')
            else:
                self.lstick_reg[fp] = l_region
                self.rstick_reg[fp] = r_region
                self._find_sticktip(fp,stick='l',hand='l')
                self._find_sticktip(fp,stick='r',hand='r')
    
    def _merge_box(self,fpos):
        lreg = self._get_box(fpos,'l')
        rreg = self._get_box(fpos,'r')
        mreg = np.logical_or(lreg,rreg)
        return mreg

    def _get_box(self,fpos,hand='l'):
        if hand=='l':
            (x,y,w,h) = self.lhand_bb[fpos,:]
        if hand=='r':
            (x,y,w,h) = self.rhand_bb[fpos,:]
        reg = np.zeros(self.frame_shape,dtype=bool)
        reg[y:y+h,x:x+w] = True
        return reg

    def _find_sticktip(self,fpos,stick='l',hand='l'):
        '''pick farthest point in stick reg from hand pos'''
        if stick == 'l':
            reg = self.lstick_reg[fpos]
        if stick == 'r':
            reg = self.rstick_reg[fpos]
        if hand == 'l':
            pos = self.lpos[fpos,:].astype(int)
            bb = self.lhand_bb[fpos,:]
        if hand == 'r':
            pos = self.rpos[fpos,:].astype(int)
            bb = self.rhand_bb[fpos,:]

        if not np.any(reg):
            tip_x, tip_y = pos
        else:
            l = self.config.sticklen
            map_disp = [l-(pos[1]-bb[1]),l-(pos[0]-bb[0])]
            map_end = [l+1+(bb[1]+bb[3]-pos[1]),l+1+(bb[0]+bb[2]-pos[0])]
            shiftedmap = self.distance_map[map_disp[0]:map_end[0],map_disp[1]:map_end[1]]

            cropreg = reg[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
            distance = np.where(cropreg,shiftedmap,0)
            tip_y, tip_x = np.unravel_index(np.argmax(distance),distance.shape)
            tip_y, tip_x = (tip_y+bb[1],tip_x+bb[0])

        if stick=='l':
            self.lstick_pos[fpos,:] = [tip_x,tip_y]
        if stick=='r':
            self.rstick_pos[fpos,:] = [tip_x,tip_y]
    
    def _flying(self,stick,start,end):
        '''stick:'l' or 'r', start/end: frame
        requires [start-1] being done
        linearly predict the stick pos, and find the region at the proximity'''
        def _closest(reg,pos):
            lab, lnum = ndi.label(reg)
            if lnum==0:
                return reg, pos
            com = ndi.center_of_mass(reg,lab,np.arange(1,lnum+1))
            disp = np.array([np.array(regp) for regp in com])
            disp = disp - np.expand_dims([pos[1],pos[0]],axis=0)
            d = np.linalg.norm(disp,axis=1)
            ix = d.argmin()
            chosen = ix+1
            chosenreg = (lab==chosen)
            chosenpos = np.array([com[ix][1],com[ix][0]],dtype=int)
            return chosenreg,chosenpos

        if stick=='l':
            pre_pos = self.lstick_pos[start-1,:]
        elif stick=='r':
            pre_pos = self.rstick_pos[start-1,:]
        
        pre_vel = np.array([0.0,0.0])

        for fp in range(start,end):
            exp_pos = pre_pos+pre_vel
            bb = CtoBsingle(exp_pos,self.config.searchlen,self.config.searchlen,self.frame_shape)
            reg = self.cleaned_region[fp][bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
            exp_pos_inner = exp_pos - bb[0:2]
            chosenreg_inner, chosenpos_inner = _closest(reg,exp_pos_inner)
            newreg = np.zeros(self.frame_shape,dtype=bool)
            newreg[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]] = chosenreg_inner
            newpos = chosenpos_inner + bb[0:2]

            if stick=='l':
                self.lstick_pos[fp,:] = newpos
                self.lstick_reg[fp] = newreg
                self.lsearch_bb[fp,:] = bb
            elif stick=='r':
                self.rstick_pos[fp,:] = newpos
                self.rstick_reg[fp] = newreg
                self.rsearch_bb[fp,:] = bb
            
            pre_vel = newpos - pre_pos
            pre_pos = newpos
    
    def _onehand(self,stick,hand,start,end):
        if hand=='l':
            hbb = self.lhand_bb[start:end,:]
        if hand=='r':
            hbb = self.rhand_bb[start:end,:]
        if stick=='l':
            self.lsearch_bb[start:end,:] = hbb
        if stick=='r':
            self.rsearch_bb[start:end,:] = hbb

        for fp in range(start,end):
            if stick=='l':
                exclude_reg = self.rstick_reg[fp]
            elif stick=='r':
                exclude_reg = self.lstick_reg[fp]

            reg = np.logical_and(self.cleaned_region[fp],~exclude_reg)

            if hand=='l':
                region = np.logical_and(reg,self._get_box(fp,'l'))
            elif hand=='r':
                region = np.logical_and(reg,self._get_box(fp,'r'))
            
            if stick=='l':
                self.lstick_reg[fp]=region
            elif stick=='r':
                self.rstick_reg[fp] = region

            self._find_sticktip(fp,stick=stick,hand=hand)





class StickDetectionWidget(QWidget):
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




class StickDetectionOperation(Operation):
    def __init__(self,res,ld):
        super().__init__(res,ld)
        self.wid = StickDetectionWidget()
        self.calc = StickDetection()
        self.hsv_loader = HsvLoader()
        self.hsv_loader.set_loader(self.ld)

    def run(self):
        self.wid.update_signal().connect(self._update_config)

        self.wid.set_config(self.calc.get_config())

        self.hsv_loader.calc()

        unit = self.res.get_unit('tracking')
        track_df = unit.get()
        lbox = track_df.loc[:,('l_x','l_y','l_w','l_h')].values
        rbox = track_df.loc[:,('r_x','r_y','r_w','r_h')].values
        self.lbool = np.any(lbox!=0,axis=1)
        self.rbool = np.any(rbox!=0,axis=1)
        stick_unit = self.res.get_unit('basics')
        d = stick_unit.get()
        ch = d['stickpos']
        stpos = StickPosition(self.ld.getframenum())
        stpos.loadchanges(ch['frame'],ch['left'],ch['right'])

        self.calc.set(self.hsv_loader,lbox,rbox,self.lbool,self.rbool,stpos)
        self.calc.main()

        self._set_view()

    def post_finish(self):
        self.wid.update_signal().disconnect(self._update_config)

        resdf = self.calc.get_pos_df()
        unit=self.res.get_unit('stick').add_df(resdf)

        regs = self.calc.get_reg()
        self.res.get_unit('left_stick').add_list(regs[0])
        self.res.get_unit('right_stick').add_list(regs[1])

    def finish_signal(self):
        return self.wid.finish_signal()

    def get_widget(self):
        return self.wid

    def viewer_setting(self,viewerset:ViewerSet):
        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':2})
        self.viewerset.deploy('grid',rows = {'single':1},cols={'single':2},order={'single':'cols'})
        self.viewers = self.viewerset.get_viewers()['single']
        for viewer in self.viewers:
            viewer.set_loader(self.hsv_loader)
            viewer.setting.enable_roi = True
            viewer.setting.show_roi_bgr = True
            viewer.apply_setting()

            drawing = viewer.get_drawing()
            drawing.vis_off()

        self.viewers[0].link_frame(self.viewers[1])
    
    def _update_config(self,config_dict):
        self.calc.set_config(config_dict)
        self.calc.main()
        self._set_view()
    
    def _set_view(self):
        lpos,rpos = self.calc.get_pos_array()
        ldraw = self.viewers[0].get_drawing()
        ldraw.set_positions({'l':lpos})
        rdraw = self.viewers[1].get_drawing()
        rdraw.set_positions({'r':rpos})

        lreg,rreg = self.calc.get_reg()
        self.viewers[0].set_masks(lreg)
        self.viewers[1].set_masks(rreg)

        lbb,rbb = self.calc.get_bb()
        self.viewers[0].set_cropbox(lbb,self.lbool)
        self.viewers[1].set_cropbox(rbb,self.rbool)

        self.viewers[0].change_fpos(0)

