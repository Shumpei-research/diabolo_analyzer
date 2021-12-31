import numpy as np
import cv2
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from operation_base import Operation
from utilities import StickPosition




class DiaboloNumberInput(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.dnum_lab = QLabel('Number of Diabolo:')
        self.dnum_edit = QLineEdit()
        self.dnum_current = QLabel('current: 1')
        self.current_dnum = 1
        self.dnum_edit.returnPressed.connect(self._set_dnum)

        self.l0.addWidget(self.dnum_lab)
        self.l0.addWidget(self.dnum_edit)
        self.l0.addWidget(self.dnum_current)
    
    def get_dnum(self):
        return self.current_dnum
    
    def _set_dnum(self):
        n_str = self.dnum_edit.text()
        if not n_str.isnumeric():
            return
        self.current_dnum = int(n_str)
        self.dnum_current.setText(f'current: {self.current_dnum}')


class StickPositionWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.current_array = np.array([[0,0,0]],dtype=int)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.description = QLabel('1:left hand, 2:right hand, 3:flying, 0:absent')
        self.l0.addWidget(self.description)

        self.table = QTableWidget(1,3)
        header = ['frame','left stick','right stick']
        self.table.setHorizontalHeaderLabels(header)
        for i in range(3):
            self.table.setItem(0,i,QTableWidgetItem(self.current_array[0,i]))

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

        self.addbutton.clicked.connect(self._addrow)
        self.delbutton.clicked.connect(self._delrow)
        self.button.clicked.connect(self._check)
    
    def get_stickpos_array(self):
        return self.current_array

    def _addrow(self):
        current = self.table.rowCount()
        self.table.setRowCount(current+1)
    def _delrow(self):
        current = self.table.rowCount()
        self.table.setRowCount(current-1)
    def _invalid_error(self):
        self.label.setText('invalid input')
    def _to_array(self):
        arr = np.zeros((self.table.rowCount(),self.table.columnCount()),dtype=int)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = self.table.item(i,j).text()
                if not txt.isdigit():
                    self._invalid_error()
                    return None
                val = int(txt)
                if j in [1,2]:
                    if val not in [0,1,2,3]:
                        self._invalid_error()
                        return None
                arr[i,j]=val
        self.label.setText('')
        return arr
    def _check(self):
        arr = self._to_array()
        if arr is None:
            return
        self.current_array = arr


class BasicInfoOperationWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.l0 = QVBoxLayout()
        self.setLayout(self.l0)

        self.dnum_wid = DiaboloNumberInput(self)
        self.l0.addWidget(self.dnum_wid)

        self.stickpos_wid = StickPositionWidget(self)
        self.l0.addWidget(self.stickpos_wid)

        self.fin_button = QPushButton('finish')
        self.l0.addWidget(self.fin_button)
    
    def finish_signal(self):
        return self.fin_button.clicked
    
    def get_dnum(self):
        return self.dnum_wid.get_dnum()
    
    def get_stickpos_array(self):
        return self.stickpos_wid.get_stickpos_array()


class BasicInfoOperation(Operation):
    '''dia number and stick position'''
    def __init__(self,res,ld):
        '''Results and Loader objects will be registered for .finish() method.'''
        super().__init__(res,ld)

        self.wid = BasicInfoOperationWidget()

    def run(self):
        '''perform calculation/interactive operation'''
        self.viewer.change_fpos(0)

    def post_finish(self):
        '''will be called after finishing operation to take out data.'''
        dnum = self.wid.get_dnum()
        stps_array = self.wid.get_stickpos_array()
        stps = StickPosition(self.ld.getframenum())
        stps.loadchanges_array(stps_array)
        chpos,left,right=stps.get_changes()
        chd = {'frame':chpos,'left':left,'right':right}
        res_dict = {'ndia':dnum, 'stickpos':chd}
        unit = self.res.get_unit('basics')
        unit.update(res_dict)

    def finish_signal(self):
        '''returns pyqtSignal that will be emited upon finish'''
        return self.wid.finish_signal()
    def get_widget(self):
        '''returns QWidget for this operation'''
        return self.wid
    def viewer_setting(self,viewerset):
        '''set viewer'''
        self.viewerset = viewerset
        self.viewerset.generate_viewers({'single':1})
        self.viewerset.deploy('single')
        self.viewer = self.viewerset.get_viewers()['single'][0]

        self.viewer.set_loader(self.ld)

        drawing = self.viewer.get_drawing()
        drawing.vis_off()

        self.viewer.setting.enable_roi = False
        self.viewer.setting.show_roi_bgr = False
        self.viewer.apply_setting()

        drawing.set_fpos(self.ld.getframenum())



