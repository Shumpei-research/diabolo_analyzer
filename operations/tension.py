import copy
import numpy as np
import pandas as pd
import scipy.optimize

import sys,os

class TensionOptimization():
    def __init__(self,chain,force,phi,tl,tr,tl_e,tr_e):
        self.string_vectors = [self._phi2vector(p[:,0],p[:,1]) for p in phi]
        self.initial_tension = self._initial_tension(chain,force,tl,tr,tl_e,tr_e)
        self.chain = chain
        self.force = force
        self.optimized,self.optimized_force,\
        self.new_tl,self.new_tr,self.new_torque = self._optimize()
    def get(self):
        return self.optimized, self.optimized_force,\
        self.new_tl,self.new_tr,self.new_torque
    def _optimize(self):
        out = copy.deepcopy(self.initial_tension)
        out_force = copy.deepcopy(self.force)
        dianum = len(self.force)
        new_torque = np.zeros((self.force[0].shape[0],dianum))
        new_tl = np.zeros((self.force[0].shape[0],dianum))
        new_tr = np.zeros((self.force[0].shape[0],dianum))
        for i in range(len(self.initial_tension)):
            ini = self.initial_tension[i]
            if ini.shape[0]<3:
                continue
            diaix = [int(c[1:]) for c in self.chain[i][1:-1]]
            vec = np.zeros((ini.shape[0]-1,2,2))
            force = np.zeros((ini.shape[0]-1,2))
            for j,d in enumerate(diaix):
                vec[j,:,:] = self.string_vectors[d][i,:,:]
                force[j,:] = self.force[d][i,:]
            lower = [0 for k in range(ini.shape[0])]
            upper = [np.inf for k in range(ini.shape[0])]
            b = scipy.optimize.Bounds(lower,upper)
            result = scipy.optimize.minimize(self._cost,ini,(vec,force),bounds=b)

            out[i] = result.x

        for i in range(len(self.initial_tension)):
            if len(self.chain[i])<3:
                continue
            diaix = [int(c[1:]) for c in self.chain[i][1:-1]]
            for j,d in enumerate(diaix):
                new_tl[i,d] = out[i][j]
                new_tr[i,d] = out[i][j+1]
                new_torque[i,d] = new_tr[i,d]-new_tl[i,d]
            t = out[i]
            vec = np.zeros((len(diaix),2,2))
            for j,d in enumerate(diaix):
                vec[j,:,:] = self.string_vectors[d][i,:,:]
            newforce = self._t2f(t,vec)
            for j,d in enumerate(diaix):
                out_force[d][i,:] = newforce[j]
        return out,out_force,new_tl,new_tr,new_torque

    def _initial_tension(self,chain,force,tl,tr,tl_e,tr_e):
        tension = [np.zeros(0) for i in chain]
        for i,c in enumerate(chain):
            if len(c)>0:
                tension[i] = np.zeros(len(c)-1)
        for i,c in enumerate(chain):
            if len(c)<=1:
                continue
            if len(c)==2:
                tension[i][0]=0.0
                continue
            if len(c)==3:
                diaix = int(c[1][1:])
                tension[i][0] = tl[diaix][i]
                tension[i][-1] = tr[diaix][i]
                if tension[i][0]<0 and tension[i][-1]<0:
                    tension[i][0]=0
                    tension[i][-1]=0
                    continue
                if tension[i][0]<0:
                    tension[i][0]=0
                    tension[i][-1] = self._guess_opposite(
                        self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][0],'r')
                    continue
                if tension[i][-1]<0:
                    tension[i][-1]=0
                    tension[i][0] = self._guess_opposite(
                        self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][-1],'l')
                    continue
                continue

            for j in range(len(c)-3):
                key1 = c[j+1]
                key2 = c[j+2]
                diaix1 = int(key1[1:])
                diaix2 = int(key2[1:])
                tl_s = tl[diaix2][i]
                tr_s = tr[diaix1][i]
                tle_s = tl_e[diaix2][i]
                tre_s = tr_e[diaix1][i]
                ratio = tre_s /(tre_s + tle_s)
                tension[i][j+1] = tr_s*(1-ratio) + tl_s*ratio
                if tension[i][j+1]<0:
                    tension[i][j+1]=0
            diaix = int(c[1][1:])
            tension[i][0] = self._guess_opposite(
                self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][1],'l')
            diaix = int(c[-2][1:])
            tension[i][-1] = self._guess_opposite(
                self.string_vectors[diaix][i,:,:],force[diaix][i,:],tension[i][-2],'r')
        return tension

    def _phi2vector(self,phil,phir):
        vecl = np.stack([np.cos(phil),-np.sin(phil)],axis=1)
        vecr = np.stack([np.cos(phir),-np.sin(phir)],axis=1)
        vec = np.stack((vecl,vecr),axis=1)
        return vec
    
    def _guess_opposite(self,vec,force,tension,toguess):
        if toguess=='l':
            v = vec[0,:]
            v_ori = vec[1,:]
        elif toguess=='r':
            v = vec[1,:]
            v_ori = vec[0,:]
        ten = tension*v_ori
        rest = force - ten
        res = np.sum(rest*v)
        return res
    
    def _t2f(self,tension,vec):
        force = np.zeros((tension.shape[0]-1,2))
        for i in range(force.shape[0]):
            tl = tension[i]
            tr = tension[i+1]
            vecl = vec[i,0,:]
            vecr = vec[i,1,:]
            force[i,:] = tl*vecl + tr*vecr
        return force

    def _cost(self,tension,vec,force):
        current_f = self._t2f(tension,vec)
        residual = force - current_f
        cost = np.sum(residual**2)
        return cost




class TestOptimize():
    def __init__(self):
        sys.path.append(os.pardir)
        from analyzer import Results
        res = Results('../test/pro2')
        res.load()
        chain_diff = res.other.by_key('object_chain')
        framenum = res.other.by_key('frame_number')
        chain = self.tochain(chain_diff,framenum)
        dianum = res.other.by_key('dianum')
        force = [None for i in range(dianum)]
        phi = [None for i in range(dianum)]
        tl = [None for i in range(dianum)]
        tr = [None for i in range(dianum)]
        tl_e = [None for i in range(dianum)]
        tr_e = [None for i in range(dianum)]
        for i in range(dianum):
            key = 'd'+str(i)
            force[i] = res.oned.get_cols([key+'_force_x',key+'_force_y']).values
            phi[i] = res.oned.get_cols([key+'_phi0',key+'_phi1']).values
            tl[i] = res.oned.get_cols([key+'_tension_l']).values
            tr[i] = res.oned.get_cols([key+'_tension_r']).values
            tl_e[i] = res.oned.get_cols([key+'_tl_e']).values
            tr_e[i] = res.oned.get_cols([key+'_tr_e']).values
        calc = TensionOptimization(chain,force,phi,tl,tr,tl_e,tr_e)
        newtension,newforces,newtl,newtr,newtor = calc.get()
        for i in range(dianum):
            key = 'd'+str(i)
            df = pd.DataFrame(newforces[i],columns=[key+'_optforce_x',key+'_optforce_y'])
            res.oned.add_df(df)
            name = [key+'_opttl',key+'_opttr',key+'_opttorque']
            arr = np.stack((newtl[:,i],newtr[:,i],newtor[:,i]),axis=1)
            df = pd.DataFrame(arr,columns=name)
            res.oned.add_df(df)
        res.other.update('tension',newtension)
        # res.save()

    def tochain(self,diff,framenum):
        chain = []
        pf = 0
        frames = diff[0]
        chains = diff[1]
        pch = chains[0]
        for frame,ch in zip(frames,chains):
            chain += [pch for i in range(pf,frame)]
            pch = ch
            pf = frame
        ch = diff[1][-1]
        chain += [ch for i in range(pf,framenum)]
        return chain