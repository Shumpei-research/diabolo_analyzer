import numpy as np

class StickPosition():
    def __init__(self,framenum):
        '''
        leftstick: column 0, rightstick: column 1
        left:1 right:2 flying:3 absent:0

        data style:
            place (ndarray[framenum,2]):
                listing position integers. column 0 is left stick.
            changes (chpos-list[changenum],left-list[changenum],right-list[changenum]):
                chpos is when any changes occured, including frame 0.
                left/right is the position integers after each change.
            changes_array (ndarray[changenum,3]):
                combined (chpos,left,right). column 0 is chpos.
        '''
        self.loadplace(np.zeros((framenum,2),dtype=int))

    def _chpos(self):
        d = self.place[1:,:] - self.place[:-1,:]
        ch = np.any(d!=0,axis=1)
        chpos = np.nonzero(ch)[0]+1
        chpos = np.insert(chpos,0,0) # first frame
        chpos = chpos.tolist()
        return chpos

    def loadplace(self,place):
        self.place = place
        self.chpos = self._chpos()
        self.left = [self.place[p,0] for p in self.chpos]
        self.right = [self.place[p,1] for p in self.chpos]

    def _setplace(self):
        if len(self.chpos)==1:
            self.place[self.chpos[0]:,0] = self.left[0]
            self.place[self.chpos[0]:,1] = self.right[0]
            return
        for n,(i,j) in enumerate(zip(self.chpos[:-1],self.chpos[1:])):
            self.place[i:j,0] = self.left[n]
            self.place[i:j,1] = self.right[n]
        self.place[self.chpos[-1]:,0] = self.left[-1]
        self.place[self.chpos[-1]:,1] = self.left[-1]

    def loadchanges(self,chpos,left,right):
        self.chpos = chpos
        self.left = left
        self.right = right
        self._setplace()

    def loadchanges_array(self,arr):
        chpos = arr[:,0].tolist()
        left = arr[:,1].tolist()
        right = arr[:,2].tolist()
        self.loadchanges(chpos,left,right)

    def get_place(self):
        return self.place
    def get_changes(self):
        return self.chpos,self.left,self.right
    def get_array(self):
        arr = np.array([self.chpos,self.left,self.right]).T
        return arr

    def get_iter(self):
        '''utility function. 
        returns iterable (start chpos, end chpos, left, right)'''
        spos = self.chpos
        epos = np.append(self.chpos[1:],len(self.place))
        out = zip(spos, epos, self.left, self.right)
        return out

    def where(self,l,r):
        '''utility funciton.
        find when (left,right) is (l,r).
        l,r is list[int] for "OR" serch.
        returns start_frames, end_frames (ndarray)'''
        lp = np.isin(self.left,l)
        rp = np.isin(self.right,r)
        ix = np.logical_and(lp,rp)
        ix2 = np.insert(ix[:-1],0,False)
        posstart = np.array(self.chpos)[ix]
        posend = np.array(self.chpos)[ix2]
        if ix[-1]==True:
            posend = np.append(posend,self.place.shape[0])
        return posstart,posend

    def swap_frame(self):
        '''utility function.
        find when sticks are swapped.
        returns ndarray of frames when swap occured'''
        normal_s1,normal_e1 = self.where([1,3],[2])
        normal_s2,normal_e2 = self.where([1],[2,3])
        reverse_s1,reverse_e1 = self.where([2],[1,3])
        reverse_s2,reverse_e2 = self.where([2,3],[1])
        normal_s = np.unique(np.concatenate((normal_s1,normal_s2)))
        reverse_s = np.unique(np.concatenate((reverse_s1,reverse_s2)))
        normal_e = np.unique(np.concatenate((normal_e1,normal_e2)))
        reverse_e = np.unique(np.concatenate((reverse_e1,reverse_e2)))
        swap1 = normal_s[np.isin(normal_s,reverse_e)]
        swap2 = normal_s[np.isin(normal_e,reverse_s)]
        swap = np.unique(np.concatenate(swap1,swap2))
        return swap


class CenterToBox():
    def __init__(self,width=None,hight=None,frame_shape=None,box=None,center=None):
        if box is not None:
            self.set_box(box)
        if center is not None:
            self.set_center(center)
        if width is not None and hight is not None:
            self.set_size(width,hight)
        if frame_shape is not None:
            self.set_frame_shape(frame_shape)
            
    def set_center(self,center):
        '''ndarray[frame,(x,y)]'''
        self.center = center.astype(int)

    def set_box(self,input_box):
        '''ndarray[frame,(x,y,w,h)]'''
        c_x = (input_box[:,0] + (input_box[:,2]/2)).astype(int)
        c_y = (input_box[:,1] + (input_box[:,3]/2)).astype(int)
        self.center = np.stack((c_x,c_y),axis=1)

    def set_size(self,width,hight):
        self.hwid = int(width/2)
        self.hhig = int(hight/2)
    def set_frame_shape(self,frame_shape):
        self.frame_shape=frame_shape
    def calc(self):
        l = self.center[:,0]-self.hwid
        r = self.center[:,0]+self.hwid+1
        d = self.center[:,1]-self.hhig
        u = self.center[:,1]+self.hhig+1

        l = np.where(l<0,0,l)
        d = np.where(d<0,0,d)
        r = np.where(r>self.frame_shape[1],self.frame_shape[1],r)
        u = np.where(r>self.frame_shape[0],self.frame_shape[0],u)

        x,y,w,h = (l,d,r-l,u-d)
        self.out_box = np.stack((x,y,w,h),axis=1)
    def get_box(self):
        return self.out_box


def CtoBsingle(center,w_length,h_length,shape):
    '''max width = 2*length+1'''
    l = int(center[0]-w_length)
    r = int(center[0]+w_length+1)
    d = int(center[1]-h_length)
    u = int(center[1]+h_length+1)

    if l<0:
        l=0
    if d<0:
        d=0
    if r>shape[0]:
        l=shape[0]
    if u>shape[1]:
        u=shape[1]
    
    x,y,w,h = (l,d,r-l,u-d)
    return [x,y,w,h]



from PyQt5.QtCore import  Qt, pyqtSignal
from PyQt5.QtGui import QBrush,QColor
from PyQt5.QtWidgets import (
QWidget, QPushButton, QVBoxLayout, 
QLabel,QTreeWidget,QTreeWidgetItem,
)







class CustomTreeItemStr(QTreeWidgetItem):
    def readcustom(self):
        return self.text(1)
    def validatecustom(self):
        return True
class CustomTreeItemInt(QTreeWidgetItem):
    def readcustom(self):
        return int(self.text(1))
    def validatecustom(self):
        try:
            int(self.text(1))
            return True
        except:
            return False
class CustomTreeItemFloat(QTreeWidgetItem):
    def readcustom(self):
        return float(self.text(1))
    def validatecustom(self):
        try:
            float(self.text(1))
            return True
        except:
            return False
class CustomTreeItemBool(QTreeWidgetItem):
    def readcustom(self):
        return bool(self.checkState(1))
    def validatecustom(self):
        return True
class CustomTreeItemBoolCheck(QTreeWidgetItem):
    def readcustom(self):
        return bool(self.checkState(0))
    def validatecustom(self):
        return True

class ConfigEditor(QWidget):
    UpdatedConfig = pyqtSignal(dict)

    def __init__(self,parent=None):
        super().__init__(parent)

        self.l1 = QVBoxLayout()
        self.setLayout(self.l1)
        self.tree=QTreeWidget()
        self.l1.addWidget(self.tree)
        self.b1 = QPushButton('update')
        self.lb1 = QLabel()
        self.l1.addWidget(self.b1)
        self.l1.addWidget(self.lb1)

        self.b1.clicked.connect(self.onClick)

    def setdict(self,treedict:dict):
        self.d = treedict
        self.itemd = {}
        self._construct()
        
    def _construct(self):
        def rec_construct(root,di,itemd):
            for k,v in di.items():
                if isinstance(v,dict):
                    branch = QTreeWidgetItem(root)
                    branch.setText(0,k)
                    itemd[k]={}
                    rec_construct(branch,v,itemd[k])
                    branch.setExpanded(True)
                    continue
                if isinstance(v,str):
                    branch = CustomTreeItemStr(root)
                    branch.setText(1,v)
                elif isinstance(v,bool):
                    branch = CustomTreeItemBool(root)
                    branch.setData(1,Qt.CheckStateRole,Qt.Checked)
                    if not v:
                        branch.setCheckState(1,Qt.Unchecked)
                elif isinstance(v,int):
                    branch = CustomTreeItemInt(root)
                    branch.setText(1,str(v))
                elif isinstance(v,float):
                    branch = CustomTreeItemFloat(root)
                    branch.setText(1,str(v))
                branch.setText(0,k)
                branch.setFlags(branch.flags()|Qt.ItemIsEditable)
                itemd[k]=branch

        self.tree.setColumnCount(2)
        self.tree.expandAll()
        rec_construct(self.tree,self.d,self.itemd)

    def _rec_read(self,itemd,readout):
        for k,v in itemd.items():
            if isinstance(v,dict):
                readout[k]={}
                self._rec_read(v,readout[k])
                continue
            if not v.validatecustom():
                v.setBackground(0,QBrush(QColor(200,0,0,100)))
                v.setBackground(1,QBrush(QColor(200,0,0,100)))
                return False
            v.setBackground(0,QBrush())
            v.setBackground(1,QBrush())
            readout[k]=v.readcustom()
        return True
    def onClick(self):
        out = {}
        success = self._rec_read(self.itemd,out)
        if success:
            self.lb1.setText('')
            self.UpdatedConfig.emit(out)
        else:
            self.lb1.setText('invalid value(s)')



class CheckTreeEditor(ConfigEditor):
    def __init__(self,parent=None):
        super().__init__(parent)
    def construct(self):
        def rec_construct(root,di,itemd):
            for k,v in di.items():
                if isinstance(v,dict):
                    branch = QTreeWidgetItem(root)
                    branch.setData(0,Qt.CheckStateRole,Qt.Checked)
                    branch.setCheckState(0,1)
                    branch.setText(0,k)
                    itemd[k]={}
                    rec_construct(branch,v,itemd[k])
                    branch.setExpanded(True)
                    continue
                elif isinstance(v,bool):
                    branch = CustomTreeItemBoolCheck(root)
                    branch.setText(0,k)
                    branch.setData(0,Qt.CheckStateRole,Qt.Checked)
                    if not v:
                        branch.setCheckState(0,Qt.Unchecked)
                itemd[k]=branch

        self.tree.setColumnCount(1)
        self.tree.expandAll()
        rec_construct(self.tree,self.d,self.itemd)

        self.tree.itemChanged.connect(self.changed)
    def changed(self,item,column):
        if item.childCount() > 0:
            self.check_branch(item, item.checkState(0))
    def check_branch(self,item,state):
        for i in range(item.childCount()):
            item.child(i).setCheckState(0,state)










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