import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (QWidget, QPushButton,  
QLineEdit,QVBoxLayout, QHBoxLayout,
QLabel,QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import pyqtSignal,QTimer

import sys,os
sys.path.append(os.pardir)

from movieimporter import Loader
from guitools import ImageBaseWidget,ViewControlBase
from visualize import DrawTextFixedPos
from utilities import StickPosition

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





class StickPositionEditorWidget(EditorAndView):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.editor
    def initUI(self):
        super().initUI()
        self.frametxt = DrawTextFixedPos(self.viewer.pli)
        self.finishbutton = QPushButton('finish')
        self.l1.addWidget(self.finishbutton)
    def drawframe(self,fpos):
        self.frametxt.draw(f'frame: {fpos}')
    def setcvimage(self,im):
        self.viewer.setcvimage(im)
    def set_array(self,arr):
        self.editor.from_array(arr)
    def get_array(self):
        return self.editor.to_array()

class StickPositionEditorControl(ViewControlBase):
    def __init__(self,loader:Loader,stickpos:StickPosition=None):
        super().__init__()
        self.ld = loader
        self.window = StickPositionEditorWidget()
        self.fpos=0
        self.stickpos = StickPosition(self.ld.framenum)
        self.stickpos.loadchanges([0],[1,2])
        if stickpos is not None:
            self.stickpos = stickpos
        self.change_fpos(self.fpos)
        self.window.KeyPressed.connect(self.keyinterp)
    def get_window(self):
        return self.window
    def finish_signal(self):
        return self.window.finishbutton.clicked
    def get_stickpos(self):
        currentarr = self.window.get_array()
        self.stickpos.loadchanges_array(currentarr)
        return self.stickpos
    def change_fpos(self, new_fpos):
        if new_fpos not in range(self.ld.framenum):
            return
        self.window.blockSignals(True)
        self.fpos=new_fpos
        self.window.drawframe(self.fpos)
        self.window.setcvimage(self.ld.getframe(self.fpos))
        self.window.blockSignals(False)

