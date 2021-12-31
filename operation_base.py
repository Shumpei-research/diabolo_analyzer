from abc import ABC, abstractmethod
from visualize import ViewerSet

class Operation(ABC):
    def __init__(self,res,ld):
        '''Results and Loader objects will be registered for .finish() method.'''
        self.res = res
        self.ld = ld
    @abstractmethod
    def run(self):
        '''perform calculation/interactive operation'''
        pass
    @abstractmethod
    def post_finish(self):
        '''will be called after finishing operation to take out data.'''
        pass
    @abstractmethod
    def finish_signal(self):
        '''returns pyqtSignal that will be emited upon finish'''
        pass
    @abstractmethod
    def get_widget(self):
        '''returns QWidget for this operation'''
        pass
    @abstractmethod
    def viewer_setting(self,viewerset:ViewerSet):
        '''set viewer'''
        pass



class StaticOperation(ABC):
    '''not interactive'''
    def __init__(self,res,ld):
        '''Results and Loader objects will be registered for .finish() method.'''
        self.res = res
        self.ld = ld
    @abstractmethod
    def run(self):
        '''perform calculation'''
        pass
    @abstractmethod
    def post_finish(self):
        '''save etc.'''
        pass