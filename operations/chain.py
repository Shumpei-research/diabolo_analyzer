import numpy as np
import pandas as pd
import copy

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,  
QVBoxLayout, QTabWidget,QLabel,QLineEdit,QTextEdit)
from PyQt5.QtCore import pyqtSignal

import sys,os
sys.path.append(os.pardir)

from utilities import WrapStateStore,StickPosition
from movieimporter import Loader
from visualize import NewDrawing
from guitools import ImageBaseKeyControl, ViewControlBase


class ObjectOnString():
    def __init__(self,key:str,pos):
        '''pos: ndarray [framenum,2] (x,y)'''
        self.key = key
        self.pos = pos
        self.frame = np.arange(pos.shape[0])
        self.dead = np.all(pos==0,axis=1)
        self.first = np.nonzero(~self.dead)[0].min()
        self.last = np.nonzero(~self.dead)[0].max()


class Angle():
    def __init__(self,l:ObjectOnString,r:ObjectOnString,c:ObjectOnString,
            first:int,last:int,initialwrap:str):
        self.l = l
        self.r = r
        self.c = c
        self.first = first
        self.last = last
        self.wrap = WrapStateStore(self.first,self.last)
        self.wrap.set_state(initialwrap)
        self.dead = np.logical_or(self.l.dead,self.r.dead,self.c.dead)
        self.calc_angle()
    def get_keys(self):
        return [o.key for o in [self.l, self.r, self.c]]
    def get_objects(self):
        return [self.l,self.r,self.c]
    def get_wrap_diff(self):
        return self.wrap.get_diffstates()
    def get_wrapstates(self):
        '''this is first:last cropped'''
        return self.wrap.get_states()
    def get_wrapstate(self,fpos):
        return self.wrap.get_state(fpos)
    def get_phi_theta(self):
        return self.phi,self.theta
    def set_initialwrap(self,wrap:str):
        self.wrap.set_state(wrap)
        self.theta_correction()
    def auto_initialwrap(self,key):
        self.set_initialwrap('n')
        if key=='fly':
            if self.theta[self.first]>np.pi:
                wrap = 'm'
            else:
                wrap = 'n'
        elif key=='land':
            if self.theta[self.first]>np.pi:
                wrap = 'n'
            else:
                wrap = 'm'
        self.set_initialwrap(wrap)
    def autofill_wrap(self,fpos=None):
        if fpos is None:
            fpos = self.first
        self.wrap.clearafter(fpos)
        iswrap,isunwrap = self.is_crossing()
        iswrap[:fpos+1]=False
        isunwrap[:fpos+1]=False
        if np.any(iswrap):
            wf = np.nonzero(iswrap)[0]
            for i in wf:
                self.wrap.R(i)
        if np.any(isunwrap):
            uf = np.nonzero(isunwrap)[0]
            for i in uf:
                self.wrap.r(i)
        self.theta_correction()
    def edit_wrap(self,fpos:int,opekey:str):
        ope = {'R':self.wrap.R,
            'L':self.wrap.L,
            'r':self.wrap.r,
            'l':self.wrap.l,
            's':self.wrap.swap,
            'p':self.wrap.passthrough,
            'c':self.wrap.clear}
        ope[opekey](fpos)
        self.theta_correction()
    def set_firstframe(self,first:int):
        self.first = first
        self.wrap.set_firstframe(first)
        self.theta_correction()
    def set_lastframe(self,last:int):
        old = self.last
        self.last = last
        self.wrap.set_lastframe(last)
        if old<self.last:
            self.autofill_wrap(old)
        self.theta_correction()
    def theta_correction(self):
        correction = np.floor(self.rawtheta[self.first]/(2*np.pi))
        temptheta = self.rawtheta - 2*np.pi*correction
        if len(self.wrap.get_wrapnum())==0:
            a=0
        iniwrapnum = self.wrap.get_wrapnum()[0]
        self.theta = temptheta + 2*np.pi*iniwrapnum
        passfr = self.wrap.search('p')
        for fr in passfr:
            if self.theta[fr]>0:
                self.theta[fr:] -= 2*np.pi
            elif self.theta[fr]<0:
                self.theta[fr:] += 2*np.pi
        swapfr = self.wrap.search('s')
        for fr in swapfr:
            self.theta[fr:] = -self.theta[fr:]

    def calc_angle(self):
        # everything is [l,r]
        self.phi_raw = [0,0] # 0-2pi.
        self.dphi   = [0,0]
        self.up     = [0,0]
        self.down   = [0,0]
        self.phi    = [0,0] # continuous radian

        # y -1* due to decartesian v.s. opencv coords
        for i in range(2):
            if i==0:
                hx = self.l.pos[:,0]
                hy = -self.l.pos[:,1]
            else:
                hx = self.r.pos[:,0]
                hy = -self.r.pos[:,1]
            dx = self.c.pos[:,0]
            dy = -self.c.pos[:,1]
            self._calc_angle_sub(hx,hy,dx,dy,i)
        self.rawtheta = self.phi[0] - self.phi[1]
        self.theta_correction()
    def _calc_angle_sub(self,hx,hy,dx,dy,i):
        vec = np.stack((hx-dx, hy-dy),axis=1)
        arctan = np.where(vec[:,0]==0,0,np.arctan(vec[:,1]/vec[:,0]))
        phi_raw = np.where(vec[:,0]>=0,arctan,arctan+np.pi)
        self.phi_raw[i] = np.where(phi_raw>=0,phi_raw,phi_raw+2*np.pi)

        self.dphi[i] = np.zeros_like(self.phi_raw[i])
        self.dphi[i][1:] = self.phi_raw[i][1:]-self.phi_raw[i][:-1]

        up = np.nonzero(self.dphi[i]>np.pi)[0]
        self.up[i] = up[np.isin(up-1,self.dead,invert=True)]
        down = np.nonzero(self.dphi[i]<-np.pi)[0]
        self.down[i] = down[np.isin(down-1,self.dead,invert=True)]

        rot = np.zeros_like(self.phi_raw[i],dtype=int)
        for j in self.up[i]:
            rot[j:] -= 1
        for j in self.down[i]:
            rot[j:] += 1

        self.phi[i] = self.phi_raw[i] + 2*np.pi*rot

    def is_flying(self):
        logic1 = np.logical_and(-1*np.pi<self.theta,self.theta<=0)
        logic2 = np.logical_and(np.pi>=self.theta,self.theta>0)
        logic1 = logic1[self.first:self.last]
        logic2 = logic2[self.first:self.last]
        wstates = self.wrap.get_states()
        logic3 = np.array(wstates)=='m'
        logic4 = np.array(wstates)=='n'
        fly1 = np.logical_and(logic1,logic3)
        fly2 = np.logical_and(logic2,logic4)
        fly_crop = np.logical_or(fly1,fly2)
        is_flying = np.zeros_like(self.theta,dtype=bool)
        is_flying[self.first:self.last]=fly_crop
        is_touching = np.zeros_like(self.theta,dtype=bool)
        is_touching[self.first:self.last]=~fly_crop
        return is_flying,is_touching
    def is_crossing(self):
        step = np.floor(self.theta/(np.pi * 2))
        dstep = step[1:]-step[:-1]
        dstep = np.concatenate((np.zeros(1,dtype=int),dstep))
        dstep[self.first]=0
        dstep[:self.first]=0
        dstep[self.last:]=0
        is_wrapping = dstep==1
        is_unwrapping = dstep==-1
        return is_wrapping,is_unwrapping
    def takeoff_frame(self):
        fly = self.is_flying()[0]
        if np.all(~fly):
            return None
        frame = np.nonzero(fly)[0].min()
        return frame
    def landing_frame(self):
        land = self.is_flying()[1]
        if np.all(~land):
            return None
        frame = np.nonzero(land)[0].min()
        return frame

class AngleSet():
    def __init__(self):
        self.angles = []
        self.flys = []
    def get(self):
        return self.angles,self.flys
    def by_object(self,pos:str,key:str):
        ak = [angle.get_keys() for angle in self.angles]
        fk = [angle.get_keys() for angle in self.flys]
        d = {'l':0,'r':1,'c':2}
        ix = d[pos]
        aix = [keys[ix]==key for keys in ak]
        fix = [keys[ix]==key for keys in fk]
        filt_a = [self.angles[i] for i in range(len(aix)) if aix[i]]
        filt_f = [self.flys[i] for i in range(len(fix)) if fix[i]]
        new = AngleSet()
        new.add_angles(*filt_a)
        new.add_flys(*filt_f)
        return new
    def by_frame(self,frame:int):
        first_a = [angle.first for angle in self.angles]
        last_a = [angle.last for angle in self.angles]
        first_f = [angle.first for angle in self.flys]
        last_f = [angle.last for angle in self.flys]

        ix_a_first = [frame>=f for f in first_a]
        ix_a_last = [frame<f for f in last_a]
        ix_a = [f and l for f,l in zip(ix_a_first,ix_a_last)]
        ix_f_first = [frame>=f for f in first_f]
        ix_f_last = [frame<f for f in last_f]
        ix_f = [f and l for f,l in zip(ix_f_first,ix_f_last)]

        filt_a = [self.angles[i] for i in range(len(ix_a)) if ix_a[i]]
        filt_f = [self.flys[i] for i in range(len(ix_f)) if ix_f[i]]
        new = AngleSet()
        new.add_angles(*filt_a)
        new.add_flys(*filt_f)
        return new
    def add_angles(self,*angles):
        for angle in angles:
            self.angles.append(angle)
    def add_flys(self,*flys):
        for angle in flys:
            self.flys.append(angle)
    def erase_after(self,fpos):
        first_a = [angle.first for angle in self.angles]
        first_f = [angle.first for angle in self.flys]
        aix = [f<=fpos for f in first_a]
        fix = [f<=fpos for f in first_f]
        self.angles = [self.angles[i] for i in range(len(aix)) if aix[i]]
        self.flys = [self.flys[i] for i in range(len(fix)) if fix[i]]
    def next_landing_takeoff(self):
        landing_frames = [a.landing_frame() for a in self.flys]
        takeoff_frames = [a.takeoff_frame() for a in self.angles]
        nolanding = [i is None for i in landing_frames]
        notakeoff = [i is None for i in takeoff_frames]
        if all(nolanding) and all(notakeoff):
            return None,None,None
        if not all(nolanding):
            firstlanding = min(np.array(landing_frames)[~np.array(nolanding)])
            first = firstlanding
        if not all(notakeoff):
            firsttakeoff= min(np.array(takeoff_frames)[~np.array(notakeoff)])
            first = firsttakeoff
        if (not all(nolanding)) and (not all(notakeoff)):
            first = min([firstlanding,firsttakeoff])

        l_ix = np.array(landing_frames)==first
        t_ix = np.array(takeoff_frames)==first
        landings = list(np.array(self.flys)[l_ix])
        takeoffs = list(np.array(self.angles)[t_ix])
        # landings/takeoffs is [] if none
        return landings,takeoffs,first
    def get_wraps(self,fly=True):
        '''only works after by_object'''
        wraps = {}
        for angle in self.angles:
            wraps.update(angle.get_wrap_diff())
        if fly:
            for angle in self.flys:
                takeoff = angle.first
                landing = angle.last
                wraps.update({takeoff:'takeoff',landing:'landing'})
                passthrough = angle.wrap.search('p')
                for p in passthrough:
                    wraps.update({p:'p'})
        return wraps
    def get_wrapstates(self,framenum):
        '''only works after by_object'''
        wraps = ['' for i in range(framenum)]
        for angle in self.angles:
            wraps[angle.first:angle.last]=angle.get_wrapstates()
        return wraps
    def get_wrapstate(self,fpos):
        '''onlyworks after by_object and by_frame'''
        if len(self.angles)==0:
            return None
        w = self.angles[0].get_wrapstate(fpos)
        return w




class ObjectChain():
    def __init__(self,lpos,rpos,dposlist):
        self.objects = []
        self.objects.append(ObjectOnString('l',lpos))
        for i in range(len(dposlist)):
            self.objects.append(ObjectOnString('d'+str(i),dposlist[i]))
        self.objects.append(ObjectOnString('r',rpos))
        
        self.framenum = lpos.shape[0]
        # [[frame,...],[[chain],...],[[flying],...],[[absent],...]]
        self.diff_states = [[],[],[],[]]

        firsts = [o.first for o in self.objects]
        lasts = [o.last for o in self.objects]
        self.allfirst = max(firsts)
        self.alllast = min(lasts)

        self.initialize_chain()

    def initialize_chain(self):
        keys = [o.key for o in self.objects]
        self.add_diff_state([0,[],[],keys])
        self.add_diff_state([self.allfirst,keys,[],[]])
        self.add_diff_state([self.alllast,[],[],keys])
    def get_objects(self,*keys):
        obj_keys = [obj.key for obj in self.objects]
        out = []
        for key in keys:
            out.append(self.objects[obj_keys.index(key)])
        return out
    def get_keys(self):
        obj_keys = [obj.key for obj in self.objects]
        return obj_keys
    def get_state(self,frame):
        bef = [f for f in self.diff_states[0] if f<=frame]
        ix = np.argmax(bef)
        current = copy.deepcopy([self.diff_states[i][ix] for i in range(4)])
        current[0]=frame
        for i in range(3):
            current[i+1] = list(current[i+1])
        return current
    def add_diff_state(self,state):
        frame = state[0]
        if frame in self.diff_states[0]:
            ix = self.diff_states[0].index(frame)
            self.diff_states[1][ix]=list(state[1])
            self.diff_states[2][ix]=list(state[2])
            self.diff_states[3][ix]=list(state[3])
            return
        self.diff_states[0].append(state[0])
        self.diff_states[1].append(list(state[1]))
        self.diff_states[2].append(list(state[2]))
        self.diff_states[3].append(list(state[3]))
        self._sort_diff()
    def erase_after(self,fpos):
        self._sort_diff()
        for i, fr in enumerate(self.diff_states[0]):
            if fr>fpos:
                ix = i
                break
            ix = None
        if ix is None:
            return
        for i in range(4):
            self.diff_states[i] = self.diff_states[i][:ix]
        
    def get_states(self):
        self._sort_diff()
        frame,chain,fly,absent = self.diff_states
        chains = [[] for i in range(self.framenum)]
        flys = [[] for i in range(self.framenum)]
        for i,(fs,fe) in enumerate(zip(frame[:-1],frame[1:])):
            chains[fs:fe]=[chain[i] for j in range(fs,fe)]
            flys[fs:fe]=[fly[i] for j in range(fs,fe)]
        chains[frame[-1]:]=[chain[-1] for j in range(frame[-1],self.framenum)]
        flys[frame[-1]:]=[fly[-1] for j in range(frame[-1],self.framenum)]
        return chains,flys
    def get_diff_states(self):
        self._sort_diff()
        return self.diff_states
        
    def _sort_diff(self):
        cp = copy.deepcopy(self.diff_states)
        frame = cp[0]
        ix = np.argsort(frame)
        for i in range(4):
            s = [cp[i][j] for j in ix]
            cp[i] = s
        self.diff_states = cp
    def takeoff(self,frame,key):
        # repeatable
        current = self.get_state(frame)
        current[1].remove(key)
        current[2].append(key)
        self.add_diff_state(current)
    def landing(self,frame,lkey,rkey,ckey):
        # repeatable
        current = self.get_state(frame)
        current[2].remove(ckey)
        chain = current[1]
        insert_ix = chain.index(rkey)
        chain.insert(insert_ix,ckey)
        self.add_diff_state(current)

class AngleAssigner():
    def __init__(self,lpos,rpos,dposlist:list,stickpos:StickPosition):
        self.oc = ObjectChain(lpos,rpos,dposlist)
        self.dianum = len(dposlist)
        self.angles = AngleSet()
        self.framenum = lpos.shape[0]
        self.stickpos = stickpos
    
    def get_results(self):
        # # wrap_initial (frame,wraplist(d0,d1,...))
        # ini_wrap_frame = self.initial_wraps_frame
        # ini_keys_wrap_dict = self.initial_wraps_keys
        # ini_wrap_list = [ini_keys_wrap_dict['d'+str(i)] for i in range(self.dianum)]
        # # wrap_diff (wraplist (d0,d1,...))
        # wrap_diff_keys_list,wrap_diff = self.get_wrap_diff()
        # wrap_diff_list = []
        # for wrapdict in wrap_diff:
        #     out = [[],[]] #framelist,opekeylist
        #     out[0]=list(sorted(wrapdict.keys()))
        #     out[1] = [wrapdict[f] for f in out[0]]
        #     wrap_diff_list.append(out)
        # wrap states
        wrap_state_list = self.get_wrapstates_list()
        name = []
        arr = np.zeros((self.framenum,self.dianum),dtype=object)
        for i,st in enumerate(wrap_state_list):
            name.append('d'+str(i)+'_wrapstate')
            arr[:,i] = st
        wrap_df = pd.DataFrame(arr,columns=name)
        # chain_diff
        chain_diff = self.oc.get_diff_states()
        # theta,phi
        thetalist,philist = self.get_theta_phi()
        basename = ['_theta','_phi0','_phi1']
        dflist = []
        for i,(theta,phi) in enumerate(zip(thetalist,philist)):
            name = ['d'+str(i)+n for n in basename]
            arr = np.concatenate((np.expand_dims(theta,1),phi),axis=1)
            dflist.append(pd.DataFrame(arr,columns=name))
        theta_phi_df = pd.concat(dflist,axis=1)

        return wrap_df,chain_diff,theta_phi_df

    def get_wrap_diff(self):
        keylist = []
        wrapslist= []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wraps(fly=True)
            keylist.append(key)
            wrapslist.append(wraps)
        return keylist,wrapslist
    def get_theta(self):
        thetadict = {}
        for i in range(self.dianum):
            key  = 'd'+str(i)
            theta = np.zeros(self.framenum)
            angles,_ = self.angles.by_object('c',key).get()
            for angle in angles:
                theta[angle.first:angle.last]=angle.theta[angle.first:angle.last]
            thetadict[key]=theta
        return thetadict
    def get_theta_phi(self):
        thetalist = []
        philist = []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            theta = np.zeros(self.framenum)
            phi = np.zeros((self.framenum,2))
            angles,_ = self.angles.by_object('c',key).get()
            for angle in angles:
                theta[angle.first:angle.last]=angle.theta[angle.first:angle.last]
                phi[angle.first:angle.last,0]=angle.phi[0][angle.first:angle.last]
                phi[angle.first:angle.last,1]=angle.phi[1][angle.first:angle.last]
            thetalist.append(theta)
            philist.append(phi)
        return thetalist,philist

    def get_wrapstates_list(self):
        wraps_list= []
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wrapstates(self.framenum)
            wraps_list.append(wraps)
        return wraps_list
    def get_wrapstates(self):
        wraps_dict= {}
        for i in range(self.dianum):
            key  = 'd'+str(i)
            wraps = self.angles.by_object('c',key).get_wrapstates(self.framenum)
            wraps_dict[key]=wraps
        return wraps_dict
    def initialize(self,chain,wraps):
        '''example.
        chain: ['l','d1','d0','r']
        wraps: ['n','nR'] (d1=n,d0=nR)
        return False if invalid input
        '''
        # for readout
        self.initial_wraps_keys = {key:wrap for key,wrap in zip(chain[1:-1],wraps)}
        self.initial_wraps_frame = self.oc.allfirst

        if wraps==['']:
            wraps = []
        if len(chain)!=len(set(chain)):
            return False
        for c in chain:
            if c not in self.oc.get_keys():
                return False
        if chain[0]!='l':
            return False
        if chain[-1]!='r':
            return False
        if len(chain)-2 != len(wraps):
            return False

        fly = self.oc.get_keys()
        for key in chain:
            fly.remove(key)
        state = [self.oc.allfirst,chain,fly,[]]
        self.oc.add_diff_state(state)
        self.oc_to_angles(self.oc.allfirst)
        for i,c in enumerate(chain[1:-1]):
            wrap = wraps[i]
            angle,_ = self.angles.by_object('c',c).by_frame(self.oc.allfirst).get()
            angle = angle[0]
            angle.set_initialwrap(wrap)
        return True
    
    def forward_repeat(self,fpos=None):
        if fpos is None:
            fpos = self.oc.allfirst
        framepos = fpos

        self.angles.erase_after(framepos)
        self.oc.erase_after(framepos)
        angles,flys = self.angles.by_frame(framepos).get()
        for a in angles:
            a.set_lastframe(self.framenum)
        for a in flys:
            a.set_lastframe(self.framenum)

        while framepos is not False:
            next = self.scan_forward(framepos)
            framepos = next
    
    def wrap_and_forward(self,frame,diaix,opekey):
        if frame not in range(self.oc.allfirst,self.oc.alllast):
            return False
        if 'd'+str(diaix) not in self.oc.get_keys():
            return False
        self.edit_wrap(frame,diaix,opekey)
        if opekey in ['R','L','r','l','c']:
            self.forward_repeat(frame)
        elif opekey in ['p']:
            self.forward_repeat(frame-1)

        return True

    def scan_forward(self,fpos):
        landings,takeoffs,frame = self.angles.by_frame(fpos).next_landing_takeoff()
        if frame is None:
            return False
        for land in landings:
            self.oc.landing(frame,*land.get_keys())
        for tkof in takeoffs:
            self.oc.takeoff(frame,tkof.get_keys()[2])
        self.oc_to_angles(frame)
        return frame

    def oc_to_angles(self,fpos):
        _,chain,fly,_ = self.oc.get_state(fpos)
        newkeysets = []
        if len(chain)>2:
            newkeysets = list(zip(chain[:-2],chain[2:],chain[1:-1]))
        newflysets = []
        if len(chain)>1:
            for flykey in fly:
                newflysets += list(zip(chain[:-1],chain[1:],[flykey for i in range(len(chain)-1)]))
        
        currentangles,currentflys = self.angles.by_frame(fpos).get()
        currentkeys = [angle.get_keys() for angle in currentangles]
        currentflykeys = [angle.get_keys() for angle in currentflys]

        for angle in currentangles:
            if angle.get_keys() not in newkeysets:
                angle.set_lastframe(fpos)
        for angle in currentflys:
            if angle.get_keys() not in newflysets:
                angle.set_lastframe(fpos)
        
        for keys in newkeysets:
            if keys not in currentkeys:
                l,r,c = self.oc.get_objects(*keys)
                newangle = Angle(l,r,c,fpos,self.oc.alllast,'n')
                previouswrap = self.angles.by_frame(fpos-1).by_object('c',keys[2]).get_wrapstate(fpos-1)
                if previouswrap is None:
                    newangle.auto_initialwrap('land')
                else:
                    newangle.set_initialwrap(previouswrap)
                newangle.autofill_wrap(fpos)
                self.angles.add_angles(newangle)
        for keys in newflysets:
            if keys not in currentflykeys:
                l,r,c = self.oc.get_objects(*keys)
                newangle = Angle(l,r,c,fpos,self.oc.alllast,'n')
                newangle.auto_initialwrap('fly')
                newangle.autofill_wrap(fpos)
                self.angles.add_flys(newangle)
        self.set_swap()

    def set_swap(self):
        swapframes = self.stickpos.swap_frame()
        for frame in swapframes:
            angles = self.angles.by_frame(frame).get()[0]
            for angle in angles:
                angle.edit_wrap(frame,'s')

    def edit_wrap(self,frame,diaix,opekey):
        angle,_ = self.angles.by_frame(frame).by_object('c','d'+str(diaix)).get()
        angle = angle[0]
        if opekey in ['R','L','r','l','c']:
            angle.set_lastframe(self.framenum)
            angle.edit_wrap(frame,opekey)
        elif opekey in ['p']:
            temp = self.angles.by_frame(frame-1).by_object('c','d'+str(diaix))
            _,fly = temp.by_object('l',angle.get_keys()[0]).get()
            fly = fly[0]
            fly.set_lastframe(self.framenum)
            fly.edit_wrap(frame,opekey)



class CASubInitial(QWidget):
    Entered = pyqtSignal(list,list)
    def __init__(self,parent):
        super().__init__(parent)
        self.initUI()
        self.b1.clicked.connect(self.enter)
    def initUI(self):
        self.l1 = QVBoxLayout()
        self.setLayout(self.l1)
        self.lb0 = QLabel('initial frame:',self)
        self.l1.addWidget(self.lb0)
        self.lb1 = QLabel(self)
        self.lb1.setText('keys separated by comma "l,d0,d1,r"')
        self.l1.addWidget(self.lb1)
        self.tb1 = QLineEdit(self)
        self.l1.addWidget(self.tb1,1)
        self.lb2 = QLabel('initial wrap state by comma ("nR,nL")\n from left to right. (can be (d1,d0))')
        self.l1.addWidget(self.lb2,1)
        self.tb2 = QLineEdit(self)
        self.l1.addWidget(self.tb2,1)
        self.b1 = QPushButton('enter',self)
        self.l1.addWidget(self.b1,1)

        self.lb_warn = QLabel(self)
        self.l1.addWidget(self.lb_warn,1)
    def enter(self):
        chaintxt = self.tb1.text()
        chain = chaintxt.split(',')
        wraptxt = self.tb2.text()
        wraps = wraptxt.split(',')
        self.Entered.emit(chain,wraps)

class CAsubwrap(QWidget):
    Entered = pyqtSignal(int,int,str)
    def __init__(self,parent):
        super().__init__(parent)
        self.initUI()
        self.b1.clicked.connect(self.enter)
    def initUI(self):
        self.l1 = QVBoxLayout()
        self.setLayout(self.l1)
        self.lb1 = QLabel('frame number to edit (e.g. "122")')
        self.l1.addWidget(self.lb1,1)
        self.tb1 = QLineEdit(self)
        self.l1.addWidget(self.tb1,1)
        self.lb2 = QLabel('diabolo index to edit (e.g. "0")')
        self.l1.addWidget(self.lb2,1)
        self.tb2 = QLineEdit(self)
        self.l1.addWidget(self.tb2,1)
        self.lb3 = QLabel('wrap operation key (r,R,l,L,p,c)')
        self.l1.addWidget(self.lb3,1)
        self.tb3 = QLineEdit(self)
        self.l1.addWidget(self.tb3,1)
        self.b1 = QPushButton('update chain',self)
        self.b1.autoDefault()
        self.l1.addWidget(self.b1,1)
        self.lb_warn = QLabel(self)
        self.l1.addWidget(self.lb_warn,1)
    def enter(self):
        frametxt = self.tb1.text()
        if not frametxt.isdigit():
            self.lb_warn.setText('invalid frame')
            return
        frame = int(frametxt)
        diatxt = self.tb2.text()
        if not diatxt.isdigit():
            self.lb_warn.setText('invalid diabolo index')
            return
        self.lb_warn.setText('')
        diaix = int(diatxt)
        wrapope = self.tb3.text()
        self.Entered.emit(frame,diaix,wrapope)


class ChainAssignWidgetBase(ImageBaseKeyControl):
    def __init__(self,parent=None):
        super().__init__(parent)

    def initUI(self):
        super().initUI()

        self.l1 = QVBoxLayout()
        self.l0.setStretch(0,3)
        self.l0.addLayout(self.l1,1)

        self.lb1 = QTextEdit(self)
        self.lb1.setReadOnly(True)
        self.l1.addWidget(self.lb1,1)
        self.tab = QTabWidget(self)
        self.l1.addWidget(self.tab,1)

        self.sub_i = CASubInitial(self)
        self.tab.addTab(self.sub_i,'initialize')
        self.sub_w = CAsubwrap(self)
        self.tab.addTab(self.sub_w,'wrap operation')
        self.tab.setTabEnabled(1,False)

        self.fin_b=QPushButton('finish')
        self.l1.addWidget(self.fin_b)



class ChainAssignWidget(ChainAssignWidgetBase):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.drawing = NewDrawing(self.pli)
    def show_wraps(self,keylist,wraplist):
        txt = ''
        for i in range(len(keylist)):
            key = keylist[i]
            txt += f'{key}:\n'
            d = wraplist[i]
            sortedkey = sorted(d.keys())
            for k in sortedkey:
                txt += f'   {k}: {d[k]}\n'
        self.lb1.setText(txt)
    def show_initialframe(self,frame):
        self.sub_i.lb0.setText(f'initial frame: {frame}')
    def update_subwindow_frame(self,frame):
        self.sub_w.tb1.setText(str(frame))
    def update_subwindow_diaix(self,diaix):
        self.sub_w.tb2.setText(str(diaix))


from PyQt5.QtCore import Qt

class ChainAssignControl(ViewControlBase):
    def __init__(self,loader:Loader,lpos,
            rpos,dposlist,stickpos:StickPosition):
        super().__init__()
        self.ld = loader
        self.calc = AngleAssigner(lpos,rpos,dposlist,stickpos)
        self.window = ChainAssignWidget()

        self.set_drawing(ini=True)
        # self.window.drawing.set_fpos(self.ld.framenum)
        # keys = self.calc.oc.get_keys()
        # # self.window.drawing.set_objectkeys(keys)
        # objects = self.calc.oc.get_objects(*keys)
        # pos = {}
        # for key,object in zip(keys,objects):
        #     pos[key] = object.pos
        # self.window.drawing.set_positions(pos)

        self.wrapframes = []
        self.fpos = 0

        self.change_fpos(self.calc.oc.allfirst)
        self.window.show_initialframe(self.calc.oc.allfirst)

        self.window.KeyPressed.connect(self.keyinterp)
        self.window.sub_i.Entered.connect(self.initialize)

    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.fin_b.clicked
    def get_results(self):
        return self.calc.get_results()
    def initialize(self,chain,wrap):
        success = self.calc.initialize(chain,wrap)
        if not success:
            self.window.sub_i.lb_warn.setText('invalid input')
            return
        self.calc.forward_repeat()
        self.get_wraps()
        self.set_drawing()
        self.window.sub_w.Entered.connect(self.wrapedit)
        self.window.tab.setTabEnabled(1,True)
        self.window.tab.setTabEnabled(0,False)
        self.change_fpos(self.fpos)
    def wrapedit(self,frame,diaix,opekey):
        success = self.calc.wrap_and_forward(frame,diaix,opekey)
        if not success:
            self.window.sub_w.lb_warn.setText('invalid input')
            return
        self.get_wraps()
        self.set_drawing()
        self.change_fpos(self.fpos)
    def set_drawing(self,ini=False):
        self.window.drawing.set_fpos(self.ld.framenum)
        keys = self.calc.oc.get_keys()
        objects = self.calc.oc.get_objects(*keys)
        pos = {}
        for key,object in zip(keys,objects):
            pos[key] = object.pos
        self.window.drawing.set_positions(pos)
        vis = self.window.drawing.getvis()
        for key in keys:
            vis[key+'_label']=True
        self.window.drawing.setvis(vis)
        if ini:
            return

        chain,_ = self.calc.oc.get_states()
        self.window.drawing.set_string(chain,pos)
        wraps = self.calc.get_wrapstates()
        lrpos = {key:pos[key] for key in wraps.keys()}
        self.window.drawing.set_wrap(wraps,lrpos)

        vis = self.window.drawing.getvis()
        for key in keys:
            if key[0]=='d':
                vis[key+'_wrap']=True
        self.window.drawing.setvis(vis)

    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos = new_fpos
        self.window.update_subwindow_frame(self.fpos)
        self.window.setcvimage(self.ld.getframe(self.fpos))
        self.window.drawing.update(self.fpos)
        self.window.blockSignals(False)
    def get_wraps(self):
        keylist,wraplist = self.calc.get_wrap_diff()
        self.wrapframes = []
        for wrapd in wraplist:
            self.wrapframes += list(wrapd.keys())
        self.wrapframes = list(sorted(set(self.wrapframes)))
        self.window.show_wraps(keylist,wraplist)
        self.wraplistofdict = wraplist
    def keyinterp(self, key):
        super().keyinterp(key)
        if key==Qt.Key_N:
            self.nextwrap()
        if key==Qt.Key_P:
            self.previouswrap()
    def nextwrap(self):
        temp = [f for f in self.wrapframes if f>self.fpos]
        if len(temp) == 0:
            return
        newf = min(temp)
        diaix = 0
        for i,d in enumerate(self.wraplistofdict):
            if newf in d.keys():
                diaix = i
        self.change_fpos(newf)
        self.window.update_subwindow_diaix(diaix)
    def previouswrap(self):
        temp = [f for f in self.wrapframes if f<self.fpos]
        if len(temp) == 0:
            return
        newf = max(temp)
        diaix = 0
        for i,d in enumerate(self.wraplistofdict):
            if newf in d.keys():
                diaix = i
        self.change_fpos(newf)
        self.window.update_subwindow_diaix(diaix)


class TestChain():
    '''test class'''
    def __init__(self):
        from analyzer import Results
        impath = '../test/td2.mov'
        self.ld = Loader(impath)
        path = '../test/pro2'
        self.res = Results(path)
        self.res.load()
        self.df = self.res.oned.get()
        basename = ['_savgol_x','_savgol_y']
        lname = ['l'+n for n in basename]
        lpos = self.df[lname].values
        rname = ['r'+n for n in basename]
        rpos = self.df[rname].values
        dposlist = []
        for i in range(2):
            dname = ['d'+str(i)+n for n in basename]
            dpos = self.df[dname].values
            dposlist.append(dpos)
        stickpos = StickPosition(lpos.shape[0])
        stickpos.loadchanges_array(np.array([[427,1,2]]))
        self.ca = ChainAssignControl(self.ld,lpos,rpos,dposlist,stickpos)
        self.ca.get_window().show()
        self.ca.finish_signal().connect(self.a)
    def a(self):
        (wrap_states_df,
        chain_diff,theta_phi_df) = self.ca.get_results()
        print(f'wrapstates:\n{wrap_states_df.iloc[600:605,:]}\nchain_diff:\n{chain_diff}\n \
        theta_phi_df:\n{theta_phi_df.iloc[600:605,:]}')

def test():
    app = QApplication([])
    t = TestChain()
    app.exec_()

if __name__=='__main__':
    test()