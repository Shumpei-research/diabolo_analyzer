import numpy as np
import pandas as pd

class ForceCalc():
    def __init__(self,grav,acc,phi,theta):
        self.grav = grav
        self.acc = acc
        self.phi = phi
        self.theta = theta
        self.pre_inverse_y()
        self._calc_force()
        self._calc_torque()
        self.post_inverse_y()

    def get_results(self):
        arr = np.stack((self.forcex,self.forcey,self.fnorm,self.fangle,
        self.tension_l,self.tension_r,
        self.torque,self.tl_e,self.tr_e,self.dT_e),axis=1)
        return arr

    def post_inverse_y(self):
        self.tension_l*=-1
        self.tension_r*=-1
        self.torque*=   -1
    def pre_inverse_y(self):
        # for cartesian
        self.phi = -self.phi
    def _calc_force(self):
        ''' calcualte force from string(acc - g). unit: mg'''
        self.forcex = 0
        self.forcey = 0
        ax, ay = self.acc[:,0],self.acc[:,1]
        gx, gy, gnorm = self.grav
        self.forcex = (ax - gx)/gnorm
        self.forcey = (ay - gy)/gnorm
        self.fnorm = np.linalg.norm(np.stack((self.forcex,self.forcey),axis=1),axis=1)

        fangle = np.arctan(self.forcey/self.forcex)
        fangle = np.where(self.forcex>=0,fangle,fangle+np.pi)
        self.fangle = np.where(fangle>=0,fangle,fangle+2*np.pi)
    def _calc_torque(self):
        ''' calculate torque from force and string angle.
        unit: mgR '''
        theta = np.squeeze(self.theta)
        theta1 = self.phi[:,0]-self.fangle
        theta2 = self.fangle-self.phi[:,1]

        dead = theta==0

        self.tension_l = self.fnorm * np.sin(theta2)/np.sin(theta)
        self.tension_r = self.fnorm * np.sin(theta1)/np.sin(theta)
        self.torque = self.tension_r - self.tension_l

        self.dT_e = np.abs(1/np.sin(theta/2))
        self.tl_e = np.abs(1/np.sin(theta))
        self.tr_e = np.abs(1/np.sin(theta))

        self.tension_l[dead]=0
        self.tension_r[dead]=0
        self.torque[dead]=0
        self.dT_e[dead]=0
        self.tl_e[dead]=0
        self.tr_e[dead]=0

class ForceCalcControl():
    def __init__(self,grav,acclist,philist,thetalist):
        self.grav = grav
        self.dianum = len(acclist)
        self.calc = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            self.calc[i]=ForceCalc(self.grav,acclist[i],philist[i],
                thetalist[i])
    def get_df(self):
        dflist = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            key = 'd'+str(i)
            basename = ['_force_x','_force_y','_fnorm','_fangle','_tension_l','_tension_r',
            '_torque','_tl_e','_tr_e','_dT_e']
            name = [key+n for n in basename]
            array = self.calc[i].get_results()
            dflist[i] = pd.DataFrame(array,columns=name)
        df = pd.concat(dflist,axis=1)
        return df

class TestForce():
    def __init__(self):
        dfpath = './test7/testing_5.csv'
        df = pd.read_csv(dfpath,index_col=0)
        self.dianum = 2
        grav = (-0.0034861023477654935, 0.24583765706667549, 0.24586237317168205)
        acc = [0 for i in range(self.dianum)]
        phi = [0 for i in range(self.dianum)]
        theta = [0 for i in range(self.dianum)]
        for i in range(self.dianum):
            key = 'd'+str(i)
            acc[i] = df[[key+'_ax',key+'_ay']].values
            phi[i] = df[[key+'_phi0',key+'_phi1']].values
            theta[i] = df[[key+'_theta']].values
        
        self.calc=ForceCalcControl(grav,acc,phi,theta)
        res = self.calc.get_df()
        dfnewpath = './test7/testing_6.csv'
        dfnew = pd.concat((df,res),axis=1)
        dfnew.to_csv(dfnewpath)