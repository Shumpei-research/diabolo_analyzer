import cv2
import os
import sys
from dataclasses import dataclass,asdict,field
from abc import ABC, abstractmethod
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QTextEdit,
QWidget, QPushButton,  
QLineEdit,
QVBoxLayout, QHBoxLayout,
QTabWidget,QLabel,QTreeWidget,QTreeWidgetItem,
)
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import  Qt, pyqtSignal,QTimer
from PyQt5.QtGui import QBrush,QColor,QPainter

from visualize import DrawTextFixedPos, TensionTorquePlotter


class MainTabWindowBase(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.tab = QTabWidget(self)
        self.metalayout = QVBoxLayout()
        self.setLayout(self.metalayout)
        self.metalayout.addWidget(self.tab)


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
        self.initUI()
        self.b1.clicked.connect(self.onClick)
    def setdict(self,treedict:dict):
        self.d = treedict
        self.itemd = {}
        self.construct()
        
    def construct(self):
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

    def initUI(self):
        self.l1 = QVBoxLayout()
        self.setLayout(self.l1)
        self.tree=QTreeWidget()
        self.l1.addWidget(self.tree)
        self.b1 = QPushButton('update')
        self.lb1 = QLabel()
        self.l1.addWidget(self.b1)
        self.l1.addWidget(self.lb1)
    def rec_read(self,itemd,readout):
        for k,v in itemd.items():
            if isinstance(v,dict):
                readout[k]={}
                self.rec_read(v,readout[k])
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
        success = self.rec_read(self.itemd,out)
        if success:
            self.lb1.setText('')
            self.UpdatedConfig.emit(out)
        else:
            self.lb1.setText('invalid value(s)')

class ImageBaseWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
    def initUI(self):
        self.l0 = QHBoxLayout()
        self.glw = pg.GraphicsLayoutWidget(self)
        self.setLayout(self.l0)
        self.l0.addWidget(self.glw)
        self.pli = pg.PlotItem()
        self.pli.invertY()
        self.pli.setAspectLocked()
        self.imi = pg.ImageItem()
        self.pli.addItem(self.imi)
        self.vb = self.pli.getViewBox()
        self.glw.addItem(self.pli)
    def setcvimage(self,im):
        nim = np.swapaxes(im,0,1)
        nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
        self.imi.setImage(nim)


class CircleFitBase(QWidget):
    KeyPressed = pyqtSignal(int)
    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)
    def initUI(self):
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l1 = QVBoxLayout()

        self.l0.addLayout(self.l1,3)

        self.l2 = QVBoxLayout()
        self.l0.addLayout(self.l2,1)
        self.configedit = ConfigEditor(self)
        self.l2.addWidget(self.configedit)




class ImageMaskBaseWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self.roi.sigRegionChangeFinished.connect(self._ROIchanged)
    def initUI(self):
        self.l0 = QVBoxLayout()
        self.glw = pg.GraphicsLayoutWidget(self)
        self.setLayout(self.l0)
        self.l0.addWidget(self.glw)
        self.pli = pg.PlotItem()
        self.pli.invertY()
        self.pli.setAspectLocked()
        self.imi = pg.ImageItem()
        self.pli.addItem(self.imi)
        self.glw.addItem(self.pli)
        self.vb = self.pli.getViewBox()

        self.maskim = pg.ImageItem()
        self.vb.addItem(self.maskim)
        self.maskim.setZValue(10)
        self.maskim.setOpacity(0.5)
        self.maskim.setLookupTable(np.array([[0,0,0],[0,255,255]]))
        self.maskim.setOpts(compositionMode=QPainter.CompositionMode_Plus)

        self.roi = pg.RectROI((0,0),(10,10))
        self.pli.addItem(self.roi)

        self.label = QLineEdit(self)
        self.label.setReadOnly(True)
        self.l0.addWidget(self.label)

    def setcvimage(self,im):
        self.image = im
        nim = np.swapaxes(im,0,1)
        nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
        self.imi.setImage(nim)
    def setmask(self,mask):
        mask = np.swapaxes(mask,0,1)
        self.maskim.setImage(mask)
    def _ROIchanged(self):
        x,y = self.roi.pos()
        w,h = self.roi.size()
        x,y,w,h = (int(i) for i in (x,y,w,h))
        crop = self.image[y:y+h,x:x+w,:]
        val = np.mean(crop,axis=(0,1))
        self.label.setText(f'B(H):{val[0]:.0f}, G(S):{val[1]:.0f}, R(V):{val[2]:.0f}')


class ImageBaseKeyControl(ImageBaseWidget):
    KeyPressed = pyqtSignal(int)
    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)
    def __init__(self,parent=None):
        super().__init__(parent)
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)




class ViewerBase(ImageBaseKeyControl):
    def __init__(self,parent=None):
        super().__init__(parent)
    def initUI(self):
        super().initUI()
        self.plotter = TensionTorquePlotter()
        self.l0.setStretch(0,1)
        self.l0.addWidget(self.plotter,1)



class GravityFitBase(ImageBaseKeyControl):
    def __init__(self,parent=None):
        super().__init__(parent)
    def initUI(self):
        super().initUI()
        self.l1 = QVBoxLayout()
        self.l0.setStretch(0,3)
        self.l0.addLayout(self.l1,1)

        self.tb1 = QLineEdit()
        self.lb1 = QLineEdit('"first,last"')
        self.lb1.setReadOnly(True)
        self.button1 = QPushButton('enter')
        self.lb2 = QLineEdit()
        self.lb2.setReadOnly(True)
        self.button2 = QPushButton('finish')
        self.l1.addWidget(self.tb1)
        self.l1.addWidget(self.lb1)
        self.l1.addWidget(self.button1)
        self.l1.addWidget(self.lb2)
        self.l1.addWidget(self.button2)

        self.frametxt = DrawTextFixedPos(self.pli)


class StickFinderBase(QWidget):
    KeyPressed = pyqtSignal(int)
    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)
    def initUI(self):
        self.move(100,100)
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l1 = QVBoxLayout()

        self.l0.addLayout(self.l1,3)

        self.l2 = QVBoxLayout()
        self.l0.addLayout(self.l2,1)
        self.configedit = ConfigEditor(self)
        self.l2.addWidget(self.configedit)
        self.finish_button=QPushButton('finish',self)
        self.l2.addWidget(self.finish_button)

        self.subl=[None for i in range(2)]
        self.subl[0]=QHBoxLayout()
        self.subl[1]=QHBoxLayout()
        self.l1.addLayout(self.subl[0])
        self.l1.addLayout(self.subl[1])

        self.imwids = [ImageMaskBaseWidget(self) for i in range(2)]
        self.hsvwids = [ImageMaskBaseWidget(self) for i in range(2)]
        for i in range(2):
            self.subl[i].addWidget(self.imwids[i])
            self.subl[i].addWidget(self.hsvwids[i])

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



class ROItool(QWidget):
    ROIselected = pyqtSignal(tuple)
    KeyPressed = pyqtSignal(int)
    TextEdited = pyqtSignal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)
        self.button.clicked.connect(self._ROIselected)
        self.tb_button.clicked.connect(self._txtedited)

    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)
    def _ROIselected(self):
        x,y = self.roi.pos()
        w,h = self.roi.size()
        x,y,w,h = (int(i) for i in (x,y,w,h))
        self.ROIselected.emit((x,y,w,h))
    def _txtedited(self):
        txt = self.txtbox.text()
        self.TextEdited.emit(txt)
    def initUI(self):
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)

        self.glw = pg.GraphicsLayoutWidget()
        self.pli = pg.PlotItem()
        self.pli.invertY()
        self.pli.setAspectLocked()
        self.imi = pg.ImageItem()
        self.pli.addItem(self.imi)
        self.glw.addItem(self.pli)
        self.l0.addWidget(self.glw,4)

        self.roi = pg.RectROI((0,0),(100,100))
        self.pli.addItem(self.roi)

        self.l1 = QVBoxLayout()
        self.label = QLabel(self)
        self.l1.addWidget(self.label)

        self.button = QPushButton('determine ROI')
        self.l1.addWidget(self.button)
        self.l0.addLayout(self.l1,1)

        self.txtbox = QLineEdit('l',self)
        self.l1.addWidget(self.txtbox)
        self.tb_button=QPushButton('determine target (l,r,d0,d1,d2,...)')
        self.l1.addWidget(self.tb_button)

        self.fin_button = QPushButton('Finish')
        self.l1.addWidget(self.fin_button)


class testroi(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.roi = ROItool(self)
        self.roi.ROIselected.connect(self.test)

        self.l = QVBoxLayout()
        self.setLayout(self.l)
        self.l.addWidget(self.roi)
    def test(self,roi):
        print(roi)


class EditorAndView(QWidget):
    KeyPressed = pyqtSignal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
        self._timer = QTimer()
        self._timer.timeout.connect(self.inactive_end)

    def keyPressEvent(self,e):
        super().keyPressEvent(e)
        self.KeyPressed.emit(e.key())
        self.inactive_time()
    def inactive_time(self):
        self.blockSignals(True)
        self._timer.start(10)
    def inactive_end(self):
        self.blockSignals(False)

    def initUI(self):
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.viewer = ImageBaseWidget(self)
        self.l0.addWidget(self.viewer,2)
        self.l1 = QVBoxLayout()
        self.l0.addLayout(self.l1,1)
        self.editor = StickPositionEditorBase()
        self.l1.addWidget(self.editor)


class StickPositionEditorBase(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()

        self.addbutton.clicked.connect(self.addrow)
        self.delbutton.clicked.connect(self.delrow)
        self.button.clicked.connect(self.check)

    def initUI(self):
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.description = QLabel('1:left hand, 2:right hand, 3:flying')
        self.l0.addWidget(self.description)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        header = ['frame','left stick','right stick']
        self.table.setHorizontalHeaderLabels(header)
        self.l0.addWidget(self.table)

        self.l1 = QHBoxLayout()
        self.l0.addLayout(self.l1)
        self.addbutton = QPushButton('add row')
        self.delbutton = QPushButton('delete row')
        self.l1.addWidget(self.addbutton)
        self.l1.addWidget(self.delbutton)

        self.button = QPushButton('update')
        self.l0.addWidget(self.button)

        self.label = QLineEdit('')
        self.label.setReadOnly(True)
        self.l0.addWidget(self.label)

    def from_array(self,arr):
        self.table.setRowCount(arr.shape[0])
        self.table.setColumnCount(arr.shape[1])
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = str(arr[i,j])
                self.table.setItem(i,j,QTableWidgetItem(val))
    def to_array(self):
        arr = np.zeros((self.table.rowCount(),self.table.columnCount()))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = self.table.item(i,j).text()
                if not txt.isdigit():
                    self.invalid_error()
                    return None
                val = int(txt)
                if j in [1,2]:
                    if val not in [0,1,2,3]:
                        self.invalid_error()
                        return None
                arr[i,j]=val
        self.label.setText('')
        return arr
    def check(self):
        arr = self.to_array()
        if arr is None:
            return
    def addrow(self):
        current = self.table.rowCount()
        self.table.setRowCount(current+1)
    def delrow(self):
        current = self.table.rowCount()
        self.table.setRowCount(current-1)
    def invalid_error(self):
        self.label.setText('invalid input')



class TestTable():
    '''test class'''
    def __init__(self):
        self.t = StickPositionEditorBase()
        d=np.array([[0,1,2],[49,3,1]])
        self.t.from_array(d)

class FitPlotBase(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()
    def initUI(self):
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)
        
        self.glw = [None for i in range(6)]
        self.pli = [None for i in range(6)]
        for i in range(6):
            self.glw[i] = pg.GraphicsLayoutWidget(self)
            self.l0.addWidget(self.glw[i])
            self.pli[i] = pg.PlotItem()
            self.glw[i].addItem(self.pli[i])

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


@dataclass
class ViewControlConfig():
    fast_skip:int = 5
    ff_skip:int = 25

class ViewControlBase(ABC):
    def __init__(self):
        self.view_c = ViewControlConfig()
    @abstractmethod
    def change_fpos(self,new_fpos):
        pass
    def keyinterp(self,key):
        if key==Qt.Key_L:
            self.forward()
        if key == Qt.Key_Right:
            self.forward()
        if key==Qt.Key_H:
            self.backward()
        if key == Qt.Key_Left:
            self.backward()
        if key == Qt.Key_K:
            self.fastbackward()
        if key == Qt.Key_Up:
            self.fastbackward()
        if key == Qt.Key_J:
            self.fastforward()
        if key == Qt.Key_Down:
            self.fastforward()
        if key == Qt.Key_F:
            self.ffforward()
        if key == Qt.Key_B:
            self.ffbackward()
    def forward(self):
        self.change_fpos(self.fpos+1)
    def backward(self):
        self.change_fpos(self.fpos-1)
    def fastforward(self):
        self.change_fpos(self.fpos+self.view_c.fast_skip)
    def fastbackward(self):
        self.change_fpos(self.fpos-self.view_c.fast_skip)
    def ffforward(self):
        self.change_fpos(self.fpos+self.view_c.ff_skip)
    def ffbackward(self):
        self.change_fpos(self.fpos-self.view_c.ff_skip)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # d = {'a':'apple','b':3,'c':True,
    # 'd':{'sub':'een','sub2':3.44}}
    # d2 = {'a':True,'b':False,'sub':{'c':True,'d':False}}
    # w = CheckTreeEditor()
    # w.setdict(d2)
    w = TestTable()
    w.t.show()
    # main_window = MainWindow()
    # fig = main_window.p1.get_figure_handle()
    # ax = fig.subplots(1,1)
    # ax.plot([0,1,2,3,4],[1,2,3,2,1])
    # main_window.show()
    sys.exit(app.exec_())