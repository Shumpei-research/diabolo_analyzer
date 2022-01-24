import copy
from typing import List

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QFormLayout, QVBoxLayout, QLineEdit, QComboBox,
QCheckBox,QHBoxLayout,QLabel,QPushButton,QTabWidget)
from PyQt5.QtCore import pyqtSignal

from operation_base import Operation





class Wrap():
    '''wrap operator for knot'''
    def __init__(self,k):
        self.k = k
    def n(self):
        n = 0
        for letter in self.k:
            if letter in ['m','l','r']:
                n -= 1
            elif letter in ['R','L']:
                n += 1
        return n
    def ope(self,key):
        if key=='R':
            self.R()
        elif key =='L':
            self.L()
        elif key=='r':
            self.r()
        elif key=='l':
            self.l()
        elif key=='p':
            self.p()
    def R(self):
        last = self.k[-1]
        if last=='m':
            self.k = 'n'
        elif last=='r':
            self.k = self.k[:-1]
        else:
            self.k += 'R'
    def L(self):
        last = self.k[-1]
        if last=='m':
            self.k = 'n'
        elif last=='l':
            self.k = self.k[:-1]
        else:
            self.k += 'L'
    def r(self):
        last = self.k[-1]
        if last=='n':
            self.k = 'm'
        elif last=='R':
            self.k = self.k[:-1]
        else:
            self.k += 'r'
    def l(self):
        last = self.k[-1]
        if last=='n':
            self.k = 'm'
        elif last=='L':
            self.k = self.k[:-1]
        else:
            self.k += 'l'
    def p(self):
        last = self.k[-1]
        if last=='n':
            self.k = 'm'
        elif last=='m':
            self.k = 'n'
        else:
            raise ValueError(f'cannot passthrough at {self.k}')


class Knot():
    def __init__(self,maxframe):
        self.maxframe = maxframe
        self.bf = 0 # begin frame
        self.ef = maxframe
        self.k0 = 'n' # initial knot.
        self.wrap_keys = np.array([],dtype=object) # wrap events
        self.wrap_frames = np.array([],dtype=int) # wrap frames
    def get_dict(self):
        out = {'begin_frame':self.bf,'end_frame':self.ef,'k0':self.k0,
            'frame':self.wrap_frames,'wrap':self.wrap_keys}
        return out
    def load(self,data:dict):
        self.bf = data['begin_frame']
        self.ef = data['end_frame']
        self.k0 = data['k0']
        self.wrap_frames = data['frame']
        self.wrap_keys = data['wrap']
    def get_passthrough(self):
        ix = np.nonzero(self.wrap_keys=='p')[0]
        p_pass = []
        n_pass = []
        for i in ix:
            if self.k(i)=='m':
                p_pass.append(i)
            elif self.k(i)=='n':
                n_pass.append(i)
        return p_pass,n_pass
    def get_clip_dict(self,new_bf,new_ef):
        '''exports dict as if new_bf and new_ef is used.
        useful when you wanna get only on-string frames.'''

        new_k0 = self.k(new_bf)
        ix = np.logical_and(self.wrap_frames>new_bf,self.wrap_frames<new_ef)
        new_frames = self.wrap_frames[ix]
        new_wrap = self.wrap_keys[ix]

        out = {'begin_frame':new_bf,'end_frame':new_ef,'k0':new_k0,
            'frame':new_frames,'wrap':new_wrap}
        return out

    def get_full_k_n(self):
        '''
        arr: knot(str) ndarray(maxframe)
        narr: n (int) ndarray (maxframe)
        '''
        arr = np.full(self.maxframe,fill_value='',dtype=object)
        narr = np.zeros(self.maxframe,dtype=int)
        w = Wrap(self.k0)

        frame_set = [self.bf]+self.wrap_frames.tolist()+[self.ef]
        for ix,(f,nf) in enumerate(zip(frame_set[:-1],frame_set[1:])):
            if ix>0:
                w.ope(self.wrap_keys[ix-1])
            arr[f:nf]=w.k
            narr[f:nf]=w.n()
        return arr, narr

    def k(self,t):
        """returns knot state.
        """
        w = Wrap(self.k0)
        boolix = self.wrap_frames<=t
        for key in self.wrap_keys[boolix]:
            w.ope(key)
        return w.k

    def n(self,t:int):
        """returns wrap number.
        wrap exactly at t is included.

        Args:
            t (int): frame to see
        """        
        w = Wrap(self.k0)
        boolix = self.wrap_frames<=t
        for key in self.wrap_keys[boolix]:
            w.ope(key)
        return w.n()

    def wrap(self,t:int,key:str):
        '''set wrap event [key] at frame [t].
        overwrites if exists already.'''
        if t in self.wrap_frames:
            ix = self.wrap_frames==t
            self.wrap_keys[ix]=key
            return
        ix = np.searchsorted(self.wrap_frames,t)
        self.wrap_frames = np.insert(self.wrap_frames,ix,t)
        self.wrap_keys = np.insert(self.wrap_keys,ix,key)
    def undo_wrap(self,t:int):
        if not t in self.wrap_frames:
            return
        ix = self.wrap_frames==t
        self.wrap_frames = np.delete(self.wrap_frames,ix)
        self.wrap_frames = np.delete(self.wrap_keys,ix)



class OnPeriod():
    def __init__(self,a_g:np.ndarray,reverse:bool):
        """[summary]
        Args:
            a_g (np.ndarray): [frame], geometrical angle (0-2pi)
            reverse (bool): if o1 and o2 are reversed. 2pi-a_g will be used.
        """
        self.reverse = reverse
        if reverse:
            self.a_g = - a_g
        else:
            self.a_g = a_g
        self.maxframe = a_g.shape[0] # maximum frame (total frame number)
        self.bf = 0 # begin frame
        self.ef = self.maxframe # end frame
        self.knot = Knot(self.maxframe)

    def begin(self,t):
        self.bf = t
        self.knot.bf = t
    def end(self,t):
        self.ef = t
        self.knot.ef = t
    def begin_fly(self,t):
        self.bf = t
        if self.a_g[t] <= np.pi:
            self.knot.k0 = 'n'
        elif self.a_g[t] > np.pi:
            self.knot.k0 = 'm'
    def is_flying(self,t):
        if self.knot.k(t) == 'n' and self.a_g[t] <= np.pi:
            return True
        elif self.knot.k(t) == 'm' and self.a_g[t] > np.pi:
            return True
        return False
    def landing_period(self):
        '''ndarray(bool)[frame] size:maxframe'''
        land_array = np.logical_not(self.flying_period())
        return land_array
    def flying_period(self):
        '''ndarray(bool)[frame] size:maxframe'''
        karr,narr = self.knot.get_full_k_n()
        fly1 = np.logical_and(karr=='n',self.a_g<=np.pi)
        fly2 = np.logical_and(karr=='m',self.a_g>np.pi)
        fly_array = np.logical_and(fly1,fly2)
        return fly_array
    def set_k0(self,k0):
        '''set k0 and derive m0.
        must be called after calling begin(t)'''
        self.knot.k0 = k0
    def get_knot(self,t:int):
        return self.knot.k(t)
    def get_full_knot(self):
        return self.knot.get_full_k_n()[0]

    def a(self):
        '''returns angle after correction.'''
        return self.a_g + 2*np.pi*(self.knot.get_full_k_n()[1])
    def get_fly(self):
        '''returns [ndarray] frames of fly events.
        only bf < frames <= ef'''
        angle = self.a()
        positive_fly_frame = np.nonzero(np.logical_and(angle[:-1]>np.pi, angle[1:]<=np.pi))[0]+1
        negative_fly_frame = np.nonzero(np.logical_and(angle[:-1]<=-np.pi, angle[1:]>-np.pi))[0]+1
        positive_fly_frame = np.array([pf for pf in positive_fly_frame if self.knot.k(pf)=='n'],dtype=int)
        negative_fly_frame = np.array([nf for nf in negative_fly_frame if self.knot.k(nf)=='m'],dtype=int)
        fly_frame = np.sort(np.concatenate((positive_fly_frame,negative_fly_frame),dtype=int))
        ix = np.logical_and(fly_frame>self.bf,fly_frame<=self.ef)
        fly_frame = fly_frame[ix]
        return fly_frame
    def get_land(self):
        '''returns [ndarray] frames of land events.
        only bf < frames <= ef'''
        angle = self.a()
        positive_land_frame = np.nonzero(np.logical_and(angle[:-1]<=np.pi, angle[1:]>np.pi))[0]+1
        negative_land_frame = np.nonzero(np.logical_and(angle[:-1]>-np.pi, angle[1:]<=-np.pi))[0]+1
        # passthrough is removed by this
        positive_land_frame = np.array([pf for pf in positive_land_frame if self.knot.k(pf)=='n'],dtype=int)
        negative_land_frame = np.array([nf for nf in negative_land_frame if self.knot.k(nf)=='m'],dtype=int)
        land_frame = np.sort(np.concatenate((positive_land_frame,negative_land_frame),dtype=int))
        ix = np.logical_and(land_frame>self.bf,land_frame<=self.ef)
        land_frame = land_frame[ix]
        return land_frame
    def get_passthrough(self):
        '''returns ndarray[frames] of passthrough'''
        p = np.concatenate((self.knot.get_passthrough())).astype(int)
        return p
    def get_wrap(self):
        '''returns [tuple(ndarray,ndarray)] frames of wrap and unwrap events.
        only bf < frames <= ef'''
        diffa = self.a_g[1:] - self.a_g[:-1]
        wrap_f = np.nonzero(diffa<-np.pi)[0] + 1
        unwrap_f = np.nonzero(diffa>np.pi)[0] + 1
        wrap_ix = np.logical_and(wrap_f>self.bf, wrap_f<=self.ef)
        wrap_f = wrap_f[wrap_ix]
        unwrap_ix = np.logical_and(unwrap_f>self.bf, unwrap_f<= self.ef)
        unwrap_f = unwrap_f[unwrap_ix]
        return wrap_f, unwrap_f
    def autowrap(self,t):
        '''auto wrap after t.
        at t is not included.'''
        wrap_f, unwrap_f = self.get_wrap()
        for f in wrap_f:
            if f<=t:
                continue
            self.knot.wrap(f,'R')
        for f in unwrap_f:
            if f<=t:
                continue
            self.knot.wrap(f,'r')
    def get_earliest_after(self,t):
        '''returns earliest land/fly event frame and isfly.
        only at/after t
        only bf< frame <= ef
        returns -1,False if nothing'''
        f = self.get_fly()
        f = f[f>=t]
        l = self.get_land()
        l = l[l>=t]

        if len(f)==0 and len(l)==0:
            return -1,False
        elif len(l)==0:
            fly_min = np.min(f)
            return fly_min, True
        elif len(f)==0:
            land_min = np.min(l)
            return land_min, False

        fly_min = np.min(f)
        land_min = np.min(l)
        isfly = fly_min<land_min
        if isfly:
            frame = fly_min
        else:
            frame = land_min
        return frame,isfly
    
    def set_wrap(self,t:int,key:str):
        self.knot.wrap(t,key)
    def undo_wrap(self,t:int):
        self.knot.undo_wrap(t)





class OneObject():
    def __init__(self,name:str,category:str,object_id:int):
        self.name = name
        self.category = category
        self.object_id = object_id

    def set(self,pos:np.ndarray,bool:np.ndarray):
        """
        Args:
            pos (np.ndarray): [frame:(x,y)]
            bool (np.ndarray): [frame] if exists
        """
        self.pos = pos
        self.bool = bool


class Angle():
    def __init__(self,o1:OneObject,o2:OneObject,o3:OneObject):
        """Contains OnPeriod instances when the angle is valid.

        Args:
            o1 (OneObject): object on the left (right hand)
            o2 (OneObject): object on the right (left hand)
            o3 (OneObject): center object (must be diabolo)
        """
        if o1.object_id >= o2.object_id:
            raise ValueError('o1 id must be smaller than o2 id')
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        self.bool = np.all(np.stack((self.o1.bool,self.o2.bool,self.o3.bool),axis=1),axis=1)
        self.on_periods = []
        self.a_g = self._calc_a_g()

    def get_names(self):
        '''returns left,center,right.
        Note that isreverse() is not considered.'''
        return self.o1.name,self.o3.name,self.o2.name
    def get_full_knot(self):
        '''both flying and landing are included.
        ndarray[frame] (str)'''
        out = np.full(self.a_g.shape,fill_value='',dtype=object)
        for op in self.on_periods:
            karr = op.get_full_knot()
            out[op.bf:op.ef]=karr[op.bf:op.ef]
        return out

    def _calc_a_g(self):
        vec1 = self.o1.pos[self.bool] - self.o3.pos[self.bool]
        vec2 = self.o2.pos[self.bool] - self.o3.pos[self.bool]
        vec1c = vec1[:,0] + vec1[:,1]*1j
        vec2c = vec2[:,0] + vec2[:,1]*1j
        a = np.angle(vec2c / vec1c)
        a_g = np.where(a<0,a+2*np.pi,a)
        out = np.zeros(self.bool.shape[0],dtype=float)
        out[self.bool] = a_g
        return out
    
    def _boolend_after(self,t:int):
        '''returns end frame based on self.bool'''
        ix = np.nonzero(~self.bool)[0]
        filtered = ix[ix>=t]
        if len(filtered)==0:
            return self.bool.shape[0]
        frame = np.min(filtered)
        return frame

    def begin(self,t:int,k0:str,reverse:bool):
        if self.ison(t):
            raise ValueError(f'already on at {t}')
        new = OnPeriod(self.a_g,reverse)
        new.begin(t)
        new.set_k0(k0)
        new.end(self.earliest_after(t))
        self.on_periods.append(new)
    
    def earliest_after(self,t:int) -> int:
        ''' returns earliest bf after t.
        bf==t is not included.
        returns self.bool-based maximum if not found'''
        earliest = self._boolend_after(t)
        for op in self.on_periods:
            if op.bf<=t:
                continue
            if earliest>op.bf:
                earliest = op.bf
        return earliest
    
    def end(self,t:int):
        if not self.ison(t):
            raise ValueError(f'not on at {t}')
        op = self.get_onperiod(t)
        if op.bf == t:
            self.on_periods.remove(op)
            return
        op.end(t)
    
    def ison(self,t:int) -> bool:
        for op in self.on_periods:
            if op.bf<=t and t<op.ef:
                return True
        return False
    
    def isreverse(self,t:int) -> bool:
        op = self.get_onperiod(t)
        return op.reverse
    
    def get_onperiod(self,t:int) -> OnPeriod:
        for op in self.on_periods:
            if op.bf<=t and t<op.ef:
                return op
        raise ValueError(f'not on at {t}')
    
    def get_knot(self,t:int) -> str:
        op = self.get_onperiod(t)
        return op.get_knot()

    def begin_fly(self,t,reverse:bool):
        if self.ison(t):
            raise ValueError(f'already on at {t}')
        new = OnPeriod(self.a_g,reverse)
        new.begin_fly(t)
        new.end(self.earliest_after(t))
        self.on_periods.append(new)
    
    def clear_after(self,t:int):
        '''clear all begin and end events after t.
        event exactly at t is also cleared'''
        for op in self.on_periods:
            if op.bf>=t:
                self.on_periods.remove(op)
                continue
            if op.ef>=t:
                op.end(self._boolend_after(t))
    def autowrap(self,t:int):
        '''autowrap after t.
        only one OnPeriod is affected.'''
        op = self.get_onperiod(t)
        op.autowrap(t)

    def get_earliest_after(self,t):
        '''returns earliest land/fly event frame and isfly.
        only after (not at) t 
        only bf< frame <= ef
        returns -1,False if nothing'''
        if not self.ison(t):
            raise ValueError(f'not on at {t}')
        op = self.get_onperiod(t)
        return op.get_earliest_after(t+1)
    
    def is_just_landing(self,t):
        '''return if just landing at t.
        useful for passthrough setting'''
        if not self.ison(t):
            raise ValueError(f'not on at {t}')
        op = self.get_onperiod(t)
        lframes = op.get_land()
        if t in lframes:
            return True
        else:
            return False
    
    def set_wrap(self,t:int,key:str):
        op = self.get_onperiod(t)
        op.set_wrap(t,key)
    def undo_wrap(self,t:int):
        op = self.get_onperiod(t)
        op.undo_wrap(t)
    
    def get_all_events(self):
        '''returns ndarray of combined event frames
        rev = ndarray[frame], True if reversed.'''
        rev = np.zeros((self.a_g.shape[0]),dtype=bool)
        l_f = [np.array([],dtype=int)]
        f_f = [np.array([],dtype=int)]
        p_f = [np.array([],dtype=int)]
        for op in self.on_periods:
            l_f.append(op.get_land())
            f_f.append(op.get_fly())
            p_f.append(op.get_passthrough())
            if op.reverse:
                rev[op.bf:op.ef] = True
        
        landing_frames = np.concatenate(l_f,axis=0)
        flying_frames = np.concatenate(f_f,axis=0)
        passthrough_frames = np.concatenate(p_f,axis=0)
        return landing_frames, flying_frames, passthrough_frames, rev










class ChainState():
    def __init__(self):
        self.chain =   []
        self.flying =  []
        self.absent =  []
    def ope(self,code,**kwargs):
        if code=='set':
            self.set(**kwargs)
        elif code=='land':
            self.land(**kwargs)
        elif code=='fly':
            self.fly(**kwargs)
        elif code=='appf':
            self.appear_fly(**kwargs)
        elif code=='appl':
            self.appear_land(**kwargs)
        elif code=='dis':
            self.disappear(**kwargs)
    def set(self,chain:list,flying:list,absent:list):
        self.chain[:] = chain
        self.flying[:] = flying
        self.absent[:] = absent
    def land(self,key,left):
        '''left: left side of key. (right-hand side).'''
        if not key in self.flying:
            raise ValueError(f'{key} not in flying')
        if not left in self.chain:
            raise ValueError(f'{left} not in chain')
        self.flying.remove(key)
        ix = self.chain.index(left)
        self.chain.insert(ix+1,key)
    def fly(self,key):
        if not key in self.chain:
            raise ValueError(f'{key} not in chain')
        self.chain.remove(key)
        self.flying.append(key)
    def appear_fly(self,key):
        if not key in self.absent:
            raise ValueError(f'{key} not in absent')
        self.absent.remove(key)
        self.flying.append(key)
    def appear_land(self,key,left=''):
        '''if left is '', key will be in left-most side at chain'''
        if not key in self.absent:
            raise ValueError(f'{key} not in absent')
        self.absent.remove(key)
        if left == '':
            self.chain.insert(0,key)
            return
        elif left not in self.chain:
            raise ValueError(f'{left} not in chain')
        ix = self.chain.index(left)
        self.chain.insert(ix+1,key)
    def disappear(self,key):
        if key in self.chain:
            self.chain.remove(key)
            self.absent.append(key)
            return
        if key in self.flying:
            self.flying.remove(key)
            self.absent.append(key)
            return
        raise ValueError(f'{key} not in chain nor flying')
    def todict(self):
        return {'chain':copy.deepcopy(self.chain),
            'flying':copy.deepcopy(self.flying),
            'absent':copy.deepcopy(self.absent)}
    def fromdict(self,data:dict):
        self.chain = data['chain']
        self.flying = data['flying']
        self.absent = data['absent']
        


class ObjectChain():
    def __init__(self,framenum):
        self.fnum = framenum

        self.object_names = []
        self.ch_frames = []
        self.ch_chain = []

    def initialize(self,frame,chain:list,flying:list,absent:list):
        self.object_names = chain+flying+absent
        self.ch_frames = []
        self.ch_chain = []
        self.ch_frames.append(frame)
        self.ch_chain.append({'code':'set','chain':chain,'flying':flying,'absent':absent})
    def event(self,frame,code,**kwargs):
        '''code and kwargs for ChainState.ope method'''
        if any([f>frame for f in self.ch_frames]):
            raise ValueError(f'{frame} is not the latest')
        ix = np.searchsorted(self.ch_frames,frame,side='right')
        self.ch_frames.insert(ix,frame)
        dic = {'code':code}
        dic.update(kwargs)
        self.ch_chain.insert(ix,dic)
    def clear_after(self,frame):
        '''clear at/after frame'''
        ix = np.nonzero(np.array(self.ch_frames)<frame)[0]
        self.ch_frames = np.array(self.ch_frames)[ix].tolist()
        self.ch_chain = np.array(self.ch_chain)[ix].tolist()
        
    def get_chainstate(self,frame) -> ChainState:
        out = ChainState()
        for i,f in enumerate(self.ch_frames):
            if f>frame:
                return out
            out.ope(**self.ch_chain[i])
        return out
    def get_changedict(self) -> dict:
        out = {'frame':self.ch_frames,'change':self.ch_chain}
        return out
    def load(self,datadict:dict):
        self.ch_frames = datadict['frame']
        self.ch_chain = datadict['change']
    def get_csdict(self):
        '''frame:list , csdict:list '''
        cs = ChainState()
        csdictlist = []
        for ch in self.ch_chain:
            cs.ope(**ch)
            csdictlist.append(cs.todict())
        return self.ch_frames,csdictlist




class AngleSet():
    def __init__(self,diabolos:list,sticks:list):
        '''list of OneObject'''
        self.diabolos = diabolos
        self.sticks = sticks

        ndia = len(self.diabolos)
        if self.sticks[0].object_id!=0:
            raise ValueError('right stick id must be 0')
        if self.sticks[1].object_id!=ndia+1:
            raise ValueError('right stick id must be ndia+1')
        for i,d in enumerate(self.diabolos):
            if d.object_id !=i+1:
                raise ValueError('diabolo id not consistent')

        self.angles = []
        olist = [self.sticks[0]]+self.diabolos+[self.sticks[1]]
        for d in self.diabolos:
            for lp,left in enumerate(olist[:-1]):
                if d is left:
                    continue
                for right in olist[lp+1:]:
                    if d is right:
                        continue
                    self.angles.append(Angle(left,right,d))
    def get(self,l,c,r):
        '''l,c,r: names of objects
        returns Angle and reverse(bool)'''
        for a in self.angles:
            if [a.o1.name,a.o2.name,a.o3.name] == [l,r,c]:
                reverse = False
                return a,reverse
            elif [a.o1.name,a.o2.name,a.o3.name] == [r,l,c]:
                reverse = True
                return a,reverse
        raise ValueError(f'name pair ({l},{r},{c}) is not found')
    def get_on(self,frame):
        alist = []
        revlist = []
        for a in self.angles:
            if a.ison(frame):
                alist.append(a)
                revlist.append(a.isreverse(frame))
        return alist, revlist
    def get_all_events(self):
        '''
        land_df: DataFrame, {'frame':[],'l':[],'r':[],'c':[]}
        fly_df: same
        pass_df: same
        '''
        land_events = {'frame':[],'l':[],'r':[],'c':[]}
        fly_events = {'frame':[],'l':[],'r':[],'c':[]}
        passthrough_events = {'frame':[],'l':[],'r':[],'c':[]}
        for a in self.angles:
            land,fly,passthrough,rev = a.get_all_events()
            l,center,r = a.get_names()
            for i in land:
                if rev[i]:
                    left = r
                    right = l
                else:
                    left = l
                    right = r
                land_events['frame'].append(i)
                land_events['l'].append(left)
                land_events['r'].append(right)
                land_events['c'].append(center)
            for i in fly:
                if rev[i]:
                    left = r
                    right = l
                else:
                    left = l
                    right = r
                fly_events['frame'].append(i)
                fly_events['l'].append(left)
                fly_events['r'].append(right)
                fly_events['c'].append(center)
            for i in passthrough:
                if rev[i]:
                    left = r
                    right = l
                else:
                    left = l
                    right = r
                passthrough_events['frame'].append(i)
                passthrough_events['l'].append(left)
                passthrough_events['r'].append(right)
                passthrough_events['c'].append(center)
        pass_df = pd.DataFrame.from_dict(passthrough_events)
        fly_df = pd.DataFrame.from_dict(fly_events)
        land_df = pd.DataFrame.from_dict(land_events)
        return land_df,fly_df,pass_df





class ChainAssigner():
    def __init__(self):
        self.dias = []
        self.sticks = []
        self.framenum = 0
        self.objectchain = None
        self.angleset = None

    def set(self,dpos_list,dbool_list,spos_list,sbool_list):
        ndia = len(dpos_list)
        self.dias = []
        for i in range(ndia):
            d = OneObject('d'+str(i),'diabolo',1+i)
            d.set(dpos_list[i],dbool_list[i])
            self.dias.append(d)
        rs = OneObject('r','stick',0)
        rs.set(spos_list[1],sbool_list[1])
        ls = OneObject('l','stick',1+ndia)
        ls.set(spos_list[0],sbool_list[0])
        self.sticks = [rs,ls]
        
        self.framenum = ls.pos.shape[0]
        self.objectchain = ObjectChain(self.framenum)
        self.angleset = AngleSet(self.dias,self.sticks)
    
    def set_initial_chain(self,frame,chain,knots,flying,absent):
        '''chain must be ['r',...,'l']. knots must be consistent order like ['nR','n']'''
        self.objectchain.initialize(frame,chain,flying,absent)
        for k,l,c,r in zip(knots,chain[:-2],chain[1:-1],chain[2:]):
            angle,reverse = self.angleset.get(l,c,r)
            angle.begin(frame,k,reverse)
        for f in flying:
            for l,r in zip(chain[:-1],chain[1:]):
                angle,reverse = self.angleset.get(l,f,r)
                angle.begin_fly(frame,reverse)
        
        self.search_forward(frame)
    
    def appear_fly(self,frame,key):
        angles,revlist = self.angleset.get_on(frame)
        for a in angles:
            a.clear_after(frame)
        self.objectchain.clear_after(frame)

        self.objectchain.event(frame,'appf',key=key)
        chain = self.objectchain.get_chainstate(frame).chain
        for l,r in zip(chain[:-1],chain[1:]):
            angle,reverse = self.angleset.get(l,key,r)
            angle.begin_fly(frame,reverse)

        self.search_forward(frame)
    
    def appear_land(self,frame,key,left,knot):
        angles,revlist = self.angleset.get_on(frame)
        for a in angles:
            a.clear_after(frame)
        self.objectchain.clear_after(frame)

        self.objectchain.event(frame,'appl',key=key,left=left)
        cs = self.objectchain.get_chainstate(frame)
        chain = cs.chain
        flying = cs.flying

        ix = chain.index(key)
        a,reverse = self.angleset.get(chain[ix-1],key,chain[ix+1])
        a.begin(frame,knot,reverse)

        for f in flying:
            a,reverse = self.angleset.get(chain[ix-1],f,chain[ix+1])
            a.end(frame)
            a,reverse = self.angleset.get(chain[ix-1],f,key)
            a.begin_fly(frame,reverse)
            a,reverse = self.angleset.get(key,f,chain[ix+1])
            a.begin_fly(frame,reverse)

        self.search_forward(frame)

    def _fly(self,frame,key,left,right):
        chain = self.objectchain.get_chainstate(frame).chain
        for l,r in zip(chain[:-1],chain[1:]):
            if l==left or r==right:
                continue
            angle,reverse = self.angleset.get(l,key,r)
            angle.begin_fly(frame,reverse)
        ix = chain.index(key)

        if ix>=2:
            l = chain[ix-2]
            c = chain[ix-1]
            rto = chain[ix+1]
            self._transition(frame,l,c,key,rto=rto)
        if ix<=len(chain)-3:
            lto = chain[ix-1]
            c = chain[ix+1]
            r = chain[ix+2]
            self._transition(frame,key,c,r,lto=lto)

        self.objectchain.event(frame,'fly',key=key)

    def _land(self,frame,key,left,right):
        chain = self.objectchain.get_chainstate(frame).chain
        for l,r in zip(chain[:-1],chain[1:]):
            if l==left and r==right:
                continue
            angle,reverse = self.angleset.get(l,key,r)
            angle.end(frame)

        self.objectchain.event(frame,'land',key=key,left=left)
        chain = self.objectchain.get_chainstate(frame).chain
        ix = chain.index(key)

        if ix>=2:
            l = chain[ix-2]
            c = chain[ix-1]
            rfrom = chain[ix+1]
            self._transition(frame,l,c,rfrom,rto=key)
        if ix<=len(chain)-3:
            lfrom = chain[ix-1]
            c = chain[ix+1]
            r = chain[ix+2]
            self._transition(frame,lfrom,c,r,lto=key)
    
    def _transition(self,t:int,l,c,r,lto=None,rto=None):
        ba,brev = self.angleset.get(l,c,r)
        ba.end(t)
        if not lto is None:
            aa,arev = self.angleset.get(lto,c,r)
        elif not rto is None:
            aa,arev = self.angleset.get(l,c,rto)
        k0 = ba.get_knot(t)
        aa.begin(t,k0,arev)
    
    def set_dissapear(self):
        '''set disappear event automatically using bool array of OneObject.
        Only ObjectChain will be modified. AngleSet will be unchanged.
        Dissapear event must be from flying state. Dissapearance of on-string object
        is irregular, and the following angle behavior will not be followed.'''
        for st in self.sticks:
            end = 1+np.argmax(np.nonzero(st.bool)[0])
            if st.name in self.objectchain.get_chainstate(end).absent:
                continue
            self.objectchain.event(end,'dis',key=st.name)
        for d in self.dias:
            end = 1+np.argmax(np.nonzero(d.bool)[0])
            if d.name in self.objectchain.get_chainstate(end).absent:
                continue
            self.objectchain.event(end,'dis',key=d.name)

    def search_forward(self,t):
        '''after t (not at t) autowrap, after (not at) t search earliest fly/land event,
        rearrenge Angle,
        then repeat these steps until no event is found.
        fly event (t) will not be recaptured by serarch(t), 
        since OnEvent is refreshed at t (new OnEvent with bf=t).
        fly at OnEvent.ef will not be reported again by Angle.'''
        angles,reverse_list = self.angleset.get_on(t)
        if len(angles)==0:
            self.set_dissapear()
            return
        earl_frame = np.zeros((len(angles)),dtype=int)
        isfly = np.zeros((len(angles)),dtype=bool)
        for i,a in enumerate(angles):
            a.autowrap(t)
            earl_frame[i],isfly[i] = a.get_earliest_after(t)
        
        if np.all(earl_frame==-1):
            self.set_dissapear()
            return

        fmin = np.min(earl_frame)
        indeces = np.nonzero(earl_frame==fmin)[0]
        for ix in indeces:
            a=angles[ix]
            l,c,r = a.get_names()
            if reverse_list[ix]:
                temp = l
                l = r
                r = temp
            if isfly[ix]:
                self._fly(fmin,c,l,r)
            else:
                self._land(fmin,c,l,r)

        self.search_forward(fmin)

    def set_wrap(self,t,key,wrap_key):
        '''Set wrap at t to Angle[anglekey],
        clear begin/end events at/after t for every Angle,
        call search_forward(t).
        If passthrough, it cancels the landing at t.
        '''
        angles,revlist = self.angleset.get_on(t)
        for a in angles:
            if key != a.get_names()[1]:
                continue
            if not a.is_just_landing(t):
                continue
            a.set_wrap(t,wrap_key)
        
        for a in angles:
            a.clear_after(t)
        self.objectchain.clear_after(t)
        
        self.search_forward(t)
    
    def undo_wrap(self,t,key):
        '''undo wrap at t to Angle[anglekey],
        clear begin/end events at/after t for every Angle,
        call search_forward(t).
        If passthrough, it induces the landing at t.
        '''
        angles,revlist = self.angleset.get_on(t)
        for a in angles:
            if key != a.get_names()[1]:
                continue
            if not a.is_just_landing(t):
                continue
            a.undo_wrap(t)
        
        for a in angles:
            a.clear_after(t)
        self.objectchain.clear_after(t)
        
        self.search_forward(t)

    
    def get(self):
        """[summary]

        Returns:
            cs_frame [list[int]]: chain state change event frames
            cs_dict [list[dict[str:list[str]]]]: chain state ['chain','flying','absent']
            land_df [DataFrame]: land events ['frame','l','c','r']
            fly_df [DataFrame]: fly events ['frame','l','c','r']
            pass_df [DataFrame]: passthrough events ['frame','l','c','r']
            knot [dict[str:ndarray[str (frames)]]]: all knot ('d0','d1',...)
        """
        cs_frame, cs_dict = self.objectchain.get_csdict()

        land_df,fly_df,pass_df = self.angleset.get_all_events()

        knot = {}
        for d in self.dias:
            knot[d.name] = np.full((self.framenum),fill_value='',dtype=object)

        for i in range(len(cs_frame)):
            bf = cs_frame[i]
            if i<len(cs_frame)-1:
                ef = cs_frame[i+1]
            else:
                ef = self.framenum

            chain = cs_dict[i]['chain']
            if len(chain)<3:
                continue
            for l,c,r in zip(chain[:-2],chain[1:-1],chain[2:]):
                a,rev = self.angleset.get(l,c,r)
                k = a.get_full_knot()
                knot[c][bf:ef] = k[bf:ef]

        return cs_frame, cs_dict, land_df, fly_df, pass_df, knot





# Position Bool must be continuous without blank frames. (except for begining and ending)
# This requirements should be equipped within Tracking/Circle/Smoothing operation.



class ChainWidget(QWidget):
    InitSignal = pyqtSignal(dict)
    AppearSignal = pyqtSignal(dict)

    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QVBoxLayout(self)
        hb = QHBoxLayout()
        hb.addWidget(QLabel('frame'))
        self.frame_line = QLineEdit()
        hb.addWidget(self.frame_line)
        self.l0.addLayout(hb)

        tab = QTabWidget()

        ini_wid = QWidget()
        self.ini_form_layout = QFormLayout(ini_wid)
        self.absent_line = QLineEdit()
        self.flying_line = QLineEdit()
        self.chain_line = QLineEdit()
        self.knot_line = QLineEdit()
        self.ini_enter = QPushButton('enter')
        self.ini_enter.clicked.connect(self._emit_init)
        self.ini_form_layout.addRow('absent',self.absent_line)
        self.ini_form_layout.addRow('flying',self.flying_line)
        self.ini_form_layout.addRow('chain',self.chain_line)
        self.ini_form_layout.addRow('knot',self.knot_line)
        self.ini_form_layout.addWidget(self.ini_enter)

        appear_wid = QWidget()
        self.appear_form_layout = QFormLayout(appear_wid)
        self.key_combo = QComboBox()
        self.isfly_check = QCheckBox()
        self.left_combo = QComboBox()
        self.appear_knot_line = QLineEdit()
        self.appear_enter = QPushButton('enter')
        self.appear_enter.clicked.connect(self._emit_appear)
        self.appear_form_layout.addRow('key',self.key_combo)
        self.appear_form_layout.addRow('is flying',self.isfly_check)
        self.appear_form_layout.addRow('key at left (r hand)',self.left_combo)
        self.appear_form_layout.addRow('knot',self.appear_knot_line)
        self.appear_form_layout.addWidget(self.appear_enter)

        tab.addTab(ini_wid,'initial chain')
        tab.addTab(appear_wid,'appear event')
        self.l0.addWidget(tab)

        self.fin_button=QPushButton('finish')
        self.l0.addWidget(self.fin_button)

    def set_bool(self,sbool,dbool):
        self.sbool = sbool
        self.dbool = dbool

        self.ini_frame = np.nonzero(np.logical_and(self.sbool[0],self.sbool[1]))[0].min()
        self.d_appear_frame = [np.nonzero(db)[0].min() for db in self.dbool]
    
    def set_object_names(self,names:list):
        self.object_names = names
        self.key_combo.clear()
        self.left_combo.clear()
        for n in self.object_names:
            self.key_combo.addItem(n)
            self.left_combo.addItem(n)

    def _emit_init(self):
        def clean_split(s):
            if s=='':
                return []
            else:
                return s.split(',')
        frame = int(self.frame_line.text())
        chain = clean_split(self.chain_line.text())
        knots =  clean_split(self.knot_line.text())
        flying = clean_split(self.flying_line.text())
        absent = clean_split(self.absent_line.text())
        self.InitSignal.emit({'frame':frame,'chain':chain,'knots':knots,
            'flying':flying,'absent':absent})
    def _emit_appear(self):
        frame = int(self.frame_line.text())
        key = self.key_combo.currentText()
        isfly = self.isfly_check.isChecked()
        left = self.left_combo.currentText()
        knot = self.appear_knot_line.text()
        self.AppearSignal.emit({'frame':frame,'key':key,
            'isfly':isfly,'left':left,'knot':knot})
    
    def finish_signal(self):
        return self.fin_button.clicked





class ChainOperation(Operation):
    def __init__(self,res,ld):
        self.res = res
        self.ld = ld

        self.calc = ChainAssigner()
        self.wid = ChainWidget()

    def viewer_setting(self, viewerset):
        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':1})
        self.viewerset.deploy('single')

        self.viewer = self.viewerset.get_viewers()['single'][0]
        self.viewer.set_loader(self.ld)
        self.viewer.setting.enable_roi = False
        self.viewer.setting.show_roi_bgr = False
        self.viewer.apply_setting()

        self.drawing = self.viewer.get_drawing()
        self.drawing.vis_off()

        self.ndia = self.res.get_unit('basics').get()['ndia']
        pos_df = self.res.get_unit('smoothened').get()
        obj_name = ['l','r'] + ['d'+str(i) for i in range(self.ndia)]
        posdict = {}
        knotdict = {}
        for n in obj_name:
            posdict[n] = np.stack((pos_df[n+'_x'],pos_df[n+'_y']),axis=1)
            knotdict[n] = np.full((self.ld.getframenum()),fill_value='',dtype=object)
        self.posdict = posdict
        chain = [[] for i in range(self.ld.getframenum())]
        self.drawing.set_positions(posdict)
        self.drawing.set_wrap(knotdict,posdict,vis=True)
        self.drawing.set_string(chain,posdict)

        self.viewer.change_fpos(0)

        self.wid.set_object_names(list(self.posdict.keys()))

    def run(self):
        dposlist = [self.posdict['d'+str(i)] for i in range(self.ndia)]
        dposbool = [np.any(arr!=0,axis=1) for arr in dposlist]
        sposlist = [self.posdict['l'], self.posdict['r']]
        sposbool = [np.any(arr!=0,axis=1) for arr in sposlist]
        self.calc.set(dposlist,dposbool,sposlist,sposbool)
        self.wid.set_bool(sposbool,dposbool)

        self.wid.InitSignal.connect(self._parse_init)
        self.wid.AppearSignal.connect(self._parse_appear)
    
    def _parse_init(self,input_data:dict):
        self.calc.set_initial_chain(**input_data)
        self._visualize_data()

    def _parse_appear(self,input_data:dict):
        if input_data['isfly']:
            self.calc.appear_fly(frame=input_data['frame'],key=input_data['key'])
        else:
            self.calc.appear_land(input_data['frame'],input_data['key'],
                input_data['left'],input_data['knot'])
        self._visualize_data()
        
    def get_widget(self):
        return self.wid
    def post_finish(self):
        pass
    def finish_signal(self):
        return self.wid.finish_signal()
    
    def _visualize_data(self):
        cs_frame, cs_dict, land_df, fly_df, pass_df, knot = self.calc.get()
        chain = [[] for i in range(self.ld.getframenum())]
        for f,fnext,cs in zip(cs_frame,cs_frame[1:]+[self.ld.getframenum()],cs_dict):
            chain[f:fnext] = [cs['chain'] for i in range(f,fnext)]
        self.drawing.set_string(chain,self.posdict)
        
        self.drawing.set_wrap(knot,self.posdict,vis=True)





# to write wrap and undo_wrap functions.