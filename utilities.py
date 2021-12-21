import numpy as np

class StickPosition(object):
    def __init__(self,framenum):
        '''
        leftstick: column 0, rightstick: column 1
        left:1 right:2 flying:3 absent:0
        '''
        self.loadplace(np.zeros((framenum,2),dtype=int))
    def _chpos(self):
        d = self.place[1:,:] - self.place[:-1,:]
        ch = np.any(d!=0,axis=1)
        chpos = np.nonzero(ch)[0]+1
        chpos = np.insert(chpos,0,0) # first frame
        return chpos
    def loadplace(self,place):
        self.place = place
        self.chpos = self._chpos()
        self.state = np.array([self.place[p,:] for p in self.chpos])
    def loadchanges(self,chpos,state):
        self.chpos = np.array(chpos,dtype=int)
        self.state = np.array(state,dtype=int)
        self._setplace()
    def loadchanges_array(self,arr):
        chpos = arr[:,0]
        state = arr[:,1:3]
        self.loadchanges(chpos,state)
    def _setplace(self):
        if self.chpos.size==1:
            self.place[self.chpos[0]:,:] = np.expand_dims(self.state[0],axis=0)
            return
        for n,(i,j) in enumerate(zip(self.chpos[:-1],self.chpos[1:])):
            self.place[i:j,:] = np.expand_dims(self.state[n],axis=0)
        self.place[self.chpos[-1]:,:] = np.expand_dims(self.state[-1],axis=0)
    def get(self):
        return self.place,self.chpos,self.state
    def get_array(self):
        return np.concatenate((np.expand_dims(self.chpos,axis=1),self.state),axis=1)
    def getiter(self):
        spos = self.chpos
        epos = np.append(self.chpos[1:],len(self.place))
        st = [self.state[i,:] for i in range(self.state.shape[0])]
        return zip(spos,epos,st)
    def where(self,l,r):
        lp = np.isin(self.state[:,0],l)
        rp = np.isin(self.state[:,1],r)
        ix = np.logical_and(lp,rp)
        ix2 = np.insert(ix[:-1],0,False)
        posstart = self.chpos[ix]
        posend = self.chpos[ix2]
        if ix[-1]==True:
            posend = np.append(posend,self.place.shape[0])
        return posstart,posend
    def swap_frame(self):
        normal_s1,normal_e = self.where([2],[1,3])
        normal_s2,normal_e = self.where([2,3],[1])
        normal_s = np.unique(np.concatenate((normal_s1,normal_s2)))
        if normal_s.size==1:
            return np.array([])
        return normal_s[1:]



class WrapState():
    '''
    'n' = 0 = neutral base
    'm' = 1 = recapture base
    'R' = 2 = right wrap
    'L' = 3 = left wrap
    'r' = 4 = right unwrap
    'l' = 5 = left unwrap
    '''
    d = {'n':0,'m':1,'R':2,
        'L':3,'r':4,'l':5}
    rev_d = {i:l for l,i in d.items()}

    def __init__(self):
        self.state = [0]
    def str2int(self,s):
        out = []
        for letter in s:
            out.append(WrapState.d[letter])
        return out
    def int2str(self,i):
        out = ''
        for n in i:
            out = out+WrapState.rev_d[n]
        return out
    def set_state(self,str_state):
        self.state = self.str2int(str_state)
    def get_state(self):
        return self.int2str(self.state)
    def R(self):
        last = self.state[-1]
        if last==1:
            self.state=[0]
        elif last==4:
            self.state=self.state[:-1]
        else:
            self.state.append(2)
    def L(self):
        last = self.state[-1]
        if last==1:
            self.state=[0]
        elif last==5:
            self.state=self.state[:-1]
        else:
            self.state.append(3)
    def r(self):
        last = self.state[-1]
        if last==0:
            self.state=[1]
        elif last==2:
            self.state=self.state[:-1]
        else:
            self.state.append(4)
    def l(self):
        last = self.state[-1]
        if last==0:
            self.state=[1]
        elif last==3:
            self.state=self.state[:-1]
        else:
            self.state.append(5)
    def swap(self):
        newstate = []
        for n in self.state:
            if n==0:
                newstate.append(1)
            elif n==1:
                newstate.append(0)
            elif n==2:
                newstate.append(5)
            elif n==3:
                newstate.append(4)
            elif n==4:
                newstate.append(3)
            elif n==5:
                newstate.append(2)
        self.state = newstate
    def passthrough(self):
        if self.state == [0]:
            self.state = [1]
        elif self.state == [1]:
            self.state = [0]
    def wrapnumber(self):
        num = 0
        for l in self.state:
            if l==0:
                num = 0
            elif l==1:
                num = -1
            elif l==2 or l==3:
                num = num+1
            elif l==4 or l==5:
                num = num-1
        return num


class WrapStateStore():
    operation_keys = ['R','L','r','l','s','p']
    def __init__(self,firstframe,lastframe):
        self.w = WrapState()
        self.initialstate = self.w.get_state()
        self.firstframe = firstframe
        self.lastframe = lastframe
        self.diff_states = {}
        self.backward = False

        self.ope = {'R':self.w.R,
            'L':self.w.L,
            'r':self.w.r,
            'l':self.w.l,
            's':self.w.swap,
            'p':self.w.passthrough}
    def get_states(self):
        states = self._makestates()
        return states
    def get_diffstates(self):
        return self.diff_states
    def get_state(self,fpos):
        self._goto(fpos)
        return self.w.get_state()
    def get_initialstate(self):
        return self.initialstate
    def get_wrapnum(self):
        states = self._makestates()
        temp = WrapState()
        wrapnum = np.zeros(self.lastframe-self.firstframe)
        for i,s in enumerate(states):
            temp.set_state(s)
            wrapnum[i] = temp.wrapnumber()
        return wrapnum
    def search(self,opekey):
        frames = [key for key,val in self.diff_states.items() if val == opekey]
        return sorted(frames)
    def set_state(self,str_state):
        self.initialstate=str_state
    def set_firstframe(self,first):
        self.firstframe=first
        self.diff_states = {key:item for key,item in self.diff_states.items()
            if key>=self.firstframe}
    def set_lastframe(self,last):
        self.lastframe = last
        self.diff_states = {key:item for key,item in self.diff_states.items()
            if key<self.lastframe}
    def R(self,fp):
        self.diff_states.update({fp:'R'})
    def r(self,fp):
        self.diff_states.update({fp:'r'})
    def L(self,fp):
        self.diff_states.update({fp:'L'})
    def l(self,fp):
        self.diff_states.update({fp:'l'})
    def swap(self,fp):
        self.diff_states.update({fp:'s'})
    def passthrough(self,fp):
        self.diff_states.update({fp:'p'})
    def clear(self,fp):
        self.diff_states.pop(fp,None)
    def clearafter(self,fp):
        l = [f for f in self.diff_states.keys() if f>fp]
        for f in l:
            self.clear(f)

    def _makestates(self):
        states = [self.initialstate for i 
            in range(self.firstframe,self.lastframe)]
        self.w.set_state(self.initialstate)
        for fp,opekey in sorted(self.diff_states.items()):
            self.ope[opekey]()
            states[fp-self.firstframe:] = [self.w.get_state() for i
                in range(fp,self.lastframe)]
        return states
    def _goto(self,fpos):
        self.w.set_state(self.initialstate)
        for fp,opekey in sorted(self.diff_states.items()):
            if fp>fpos:
                return
            self.ope[opekey]()