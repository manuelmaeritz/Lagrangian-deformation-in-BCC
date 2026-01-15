import numpy as np
import scipy as sc
import os
import time
from datetime import date
import pickle
from lpt import LPEnsemble
from joblib import Parallel, delayed


class DeformationTensor:

    streamline = None

    F = None

    t = None

    dt = None

    _r = None

    def __init__(self, streamline, dt):
        self.dt = dt
        self.streamline = streamline
        # Deformation tensors shall be computed at equi temporal points dt.
        # This makes taking temporal ensemble averages easier
        self.update_time()

    @property
    def r(self):
        if self._r is None or (len(self._r) < len(self.t)):
            self._r = self.streamline.r_interp(self.t)
        return self._r

    @property
    def L(self):
        F1 = self.F[:, :, 0]
        F2 = self.F[:, :, 1]
        F3 = self.F[:, :, 2]

        L3 = np.cross(F1, F2)
        L2 = -np.cross(F1, F3)
        L1 = np.cross(F2, F3)
        L = np.stack((L1, L2, L3), axis=2)
        return L

    @property
    def LTL(self):
        """ lyap: dimensionless lyapunov exponent (lyap=tau0*lyapunov (where
        lyapunov has dimension 1/s))
        """
        L = self.L
        LT = np.transpose(L, axes=(0, 2, 1))
        LTL = np.matmul(LT, L)
        return LTL

    def cal_T(self, Pe, lyap):
        """ lyap: dimensionless lyapunov exponent (lyap=tau0*lyapunov (where
        lyapunov has dimension 1/t))
        """
        tau0 = self.streamline.pm.cal_tau0()
        L = self.L
        LT = np.transpose(L, axes=(0, 2, 1))
        LTL = np.matmul(LT, L)
        T = sc.integrate.cumtrapz(LTL, x=self.t/tau0, initial=0, axis=0)
        T = 4*lyap/Pe*T
        E = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        return E+T

    def update_time(self):
        t0 = self.streamline.t[0]
        tfi = self.streamline.t[-1]
        # we need to have tfi+1 to avoid empty times in case t0=tfi
        self.t = np.arange(t0, tfi+1, self.dt)

    def cal_deformTens(self):
        r = self.streamline.r_interp(self.t)
        epsi = self.streamline.pm.get_Jv(r)
        dt = np.diff(self.t)
        F = [np.identity(3)]
        for dt_i, epsi_i in zip(dt, epsi):
            F_t = np.matmul(np.identity(3)+epsi_i*dt_i, F[-1])
            F.append(F_t)
        self._F = np.array(F[0: -1])
        return self._F

    def cal_FTF(self):
        FT = np.transpose(self.F, axes=(0, 2, 1))
        return np.matmul(FT, self.F)

    def cal_lyapunov(self):
        FTF = self.cal_FTF()
        w, _ = np.linalg.eig(FTF)
        # print(w.shape)
        # print(self._t.reshape(-1,1).shape)
        w = np.sort(w, axis=1)
        w[:, 0] = 1/w[:, 1]/w[:, 2]
        t = self.t-self.t[0]
        lamb = np.divide(np.log(w), 2*t.reshape(-1, 1))
        return np.real(lamb)


class ProteanDeformationTensor(DeformationTensor):

    _Q = None

    _FPrime = None

    _alpha = None

    def __init__(self, streamline, dt, alpha=None, Q=None):
        super().__init__(streamline, dt)
        if Q is None or alpha is None:
            self._Q = np.empty((0, 3, 3))
            self._alpha = np.empty((0))
        else:
            if len(Q) != len(alpha):
                raise ValueError("Length of Q and alpha need to match")
            self._Q = Q
            self._alpha = alpha

    @property
    def Q(self):
        if len(self.r) > len(self._Q):
            print("Streamline is ahead of deformation. Need to integrate")
            _ = self.cal_protean_rotation()
        return self._Q

    @property
    def epsiPrime(self):
        epsi = self.streamline.pm.get_Jv(self.r)
        epsiTilde = self._cal_epsiTilde(epsi, self.Q)
        return self._cal_epsiPrime(epsiTilde)

    @property
    def FPrime(self):
        if self._FPrime is None or (len(self._FPrime) < len(self.Q)):
            v = self.streamline.pm.get_v(self.r)
            self._FPrime = self._cal_FPrime(self.t, v, self.epsiPrime)
        return self._FPrime

    @property
    def F(self):
        QT = np.transpose(self.Q[0])
        F = np.tensordot(self.FPrime, QT, axes=(2, 0))
        F = np.matmul(self.Q, F)
        return F

    @property
    def LPrime(self):
        F1 = self.FPrime[:, :, 0]
        F2 = self.FPrime[:, :, 1]
        F3 = self.FPrime[:, :, 2]

        L3 = np.cross(F1, F2)
        L2 = -np.cross(F1, F3)
        L1 = np.cross(F2, F3)
        L = np.stack((L1, L2, L3), axis=2)
        return L

    @property
    def LTL(self):
        """ lyap: dimensionless lyapunov exponent (lyap=tau0*lyapunov (where
        lyapunov has dimension 1/s))
        """
        L = self.LPrime
        LT = np.transpose(L, axes=(0, 2, 1))
        LTL = np.matmul(LT, L)
        return LTL

    def cal_deformTens(self):
        pass

    def cal_protean_rotation(self, alpha0=0):
        v = self.streamline.pm.get_v(self.r)
        Q1 = self._cal_Q1(v)
        n = len(self._Q)
        epsi_add = self.streamline.pm.get_Jv(self.r[n:])
        epsi1_add = self._cal_epsi1(Q1[n:], epsi_add)
        if len(self._alpha) > 0:
            alpha0 = self._alpha[-1]
        alpha_add = self._cal_alpha(self.t[n:], v[n:], epsi1_add,
                                    alpha0)
        self._alpha = np.concatenate((self._alpha, alpha_add))
        Q2 = self._cal_Q2(self._alpha)
        self._Q = np.matmul(Q1, Q2)
        return self._Q, self._alpha

    @staticmethod
    def _cal_Q1(v):
        absV = np.sqrt(np.sum(v**2, axis=1))
        ctheta = v[:, 0]/absV
        theta = np.arccos(ctheta)
        stheta = np.sin(theta)
        v2 = v[:, 1]**2+v[:, 2]**2
        beta = (1-ctheta)/v2

        Q = np.zeros((v.shape[0], 3, 3))
        Q[:, 0, 0] = ctheta
        Q[:, 0, 1] = -stheta*v[:, 1]/np.sqrt(v2)
        Q[:, 0, 2] = -stheta*v[:, 2]/np.sqrt(v2)
        Q[:, 1, 0] = stheta*v[:, 1]/np.sqrt(v2)
        Q[:, 1, 1] = beta*v[:, 2]**2+ctheta
        Q[:, 1, 2] = -beta*v[:, 2]*v[:, 1]
        Q[:, 2, 0] = stheta*v[:, 2]/np.sqrt(v2)
        Q[:, 2, 1] = -beta*v[:, 2]*v[:, 1]
        Q[:, 2, 2] = beta*v[:, 1]**2+ctheta
        return Q

    @staticmethod
    def _cal_Q2(alpha):
        Q2 = np.zeros((len(alpha), 3, 3))
        Q2[:, 0, 0] = np.ones(len(alpha))
        Q2[:, 1, 1] = np.cos(alpha)
        Q2[:, 2, 2] = np.cos(alpha)
        Q2[:, 1, 2] = -np.sin(alpha)
        Q2[:, 2, 1] = np.sin(alpha)
        return Q2

    @staticmethod
    def _cal_epsi1(Q1, epsi):
        Q1T = np.transpose(Q1, (0, 2, 1))
        epsi1 = np.matmul(Q1T, np.matmul(epsi, Q1))
        return epsi1

    @staticmethod
    def _cal_epsiTilde(epsi, Q):
        epsi = np.array(epsi)
        QT = np.transpose(Q, (0, 2, 1))
        epsiTilde = np.matmul(QT, np.matmul(epsi, Q))
        return epsiTilde

    @staticmethod
    def _cal_epsiPrime(epsiTilde):
        epsiPrime = np.zeros((epsiTilde.shape[0], 3, 3))
        epsiPrime[:, 0, 0] = epsiTilde[:, 0, 0]
        epsiPrime[:, 0, 1] = epsiTilde[:, 0, 1] + epsiTilde[:, 1, 0]
        epsiPrime[:, 0, 2] = epsiTilde[:, 0, 2] + epsiTilde[:, 2, 0]
        epsiPrime[:, 1, 1] = epsiTilde[:, 1, 1]
        epsiPrime[:, 1, 2] = epsiTilde[:, 1, 2] + epsiTilde[:, 2, 1]
        epsiPrime[:, 2, 2] = -epsiTilde[:, 1, 1] - epsiTilde[:, 0, 0]
        return epsiPrime

    @staticmethod
    def _cal_FPrime(t, v, epsiPrime):
        absV = np.sqrt(np.sum(v**2, axis=1))
        F11 = absV/absV[0]
        intEpsi22 = sc.integrate.cumtrapz(epsiPrime[:, 1, 1], x=t, initial=0)
        F22 = np.exp(intEpsi22)
        intEpsi33 = sc.integrate.cumtrapz(epsiPrime[:, 2, 2], x=t, initial=0)
        F33 = np.exp(intEpsi33)
        F12 = absV * sc.integrate.cumtrapz(epsiPrime[:, 0, 1]*F22/absV,
                                           t, initial=0)
        F23 = F22 * sc.integrate.cumtrapz(epsiPrime[:, 1, 2]*F33/F22, x=t,
                                          initial=0)
        F13 = absV * sc.integrate.cumtrapz((epsiPrime[:, 0, 1]*F23 +
                                            epsiPrime[:, 0, 2]*F33)/absV,
                                           x=t, initial=0)
        F = np.zeros(epsiPrime.shape)
        F[:, 0, 0] = F11
        F[:, 0, 1] = F12
        F[:, 0, 2] = F13
        F[:, 1, 1] = F22
        F[:, 1, 2] = F23
        F[:, 2, 2] = F33
        return F

    @staticmethod
    def g(alpha, t, *args):
        t_supp = args[0]
        v = args[1]
        epsi1 = args[2]
        v1 = np.sqrt(np.sum(v**2, axis=1))+v[:, 0]

        a = -epsi1[:, 2, 1]-v[:, 1]/v1*epsi1[:, 2, 0]+v[:, 2]/v1*epsi1[:, 1, 0]
        b = epsi1[:, 1, 2]-v[:, 1]/v1*epsi1[:, 2, 0]+v[:, 2]/v1*epsi1[:, 1, 0]
        c = epsi1[:, 1, 1]-epsi1[:, 2, 2]

        return -np.interp(t, t_supp,
                          a*np.cos(alpha)**2.
                          + b*np.sin(alpha)**2.
                          + c*np.cos(alpha)*np.sin(alpha))

    @staticmethod
    def Dg(alpha, t, *args):
        t_supp = args[0]
        v = args[1]
        epsi1 = args[2]
        v1 = np.sqrt(np.sum(v**2, axis=1))+v[:, 0]

        a = -epsi1[:, 2, 1]-v[:, 1]/v1*epsi1[:, 2, 0]+v[:, 2]/v1*epsi1[:, 1, 0]
        b = epsi1[:, 1, 2]-v[:, 1]/v1*epsi1[:, 2, 0]+v[:, 2]/v1*epsi1[:, 1, 0]
        c = epsi1[:, 1, 1]-epsi1[:, 2, 2]

        return -np.interp(t, t_supp, (b-a)*np.sin(2*alpha)+c*np.cos(2*alpha))

    @classmethod
    def _cal_alpha(cls, t, v, epsi1, alpha0):
        alpha = sc.integrate.odeint(cls.g, alpha0, t, args=(t, v, epsi1),
                                    Dfun=cls.Dg).flatten()
        return alpha

    def _cal_A23(self):
        r = self.streamline.r_interp(self.t)
        v = self.streamline.pm.get_v(r)
        epsi = self.streamline.pm.get_Jv(r)
        alpha = self._alpha
        epsiTilde = self._cal_epsiTilde(epsi, self.Q)
        Q1 = self._cal_Q1(v)
        epsi1 = self._cal_epsi1(Q1, epsi)

        absv = np.sqrt(np.sum(v**2, axis=1))
        A23 = (v[:, 2]*np.cos(alpha)
               - v[:, 1]*np.sin(alpha))/(absv+v[:, 0])*epsiTilde[:, 1, 0]
        A23 -= (v[:, 1]*np.cos(alpha)
                + v[:, 2]*np.sin(alpha))/(absv+v[:, 0])*epsiTilde[:, 2, 0]
        dtalpha = self.g(alpha, self._t, self._t, v, epsi1)
        A23 += dtalpha
        return A23

    def _cal_A_numerically(self):
        Q = self.Q
        dt = self._t[1]-self._t[0]
        QT = np.transpose(Q, (0, 2, 1))
        dQT = (np.roll(QT, -1, axis=0)-np.roll(QT, 1, axis=0))/2/dt
        dQT[0] = dQT[1]
        dQT[-1] = dQT[-2]
        A = np.matmul(dQT, Q)
        return A

    def interp_epsiPrime22(self, t):
        return np.interp(t, self.t, self.epsiPrime[:, 1, 1], left=np.nan,
                         right=np.nan)

    def interp_epsiPrime12(self, t):
        return np.interp(t, self.t, self.epsiPrime[:, 0, 1], left=np.nan,
                         right=np.nan)

    def interp_FPrime12(self, t):
        return np.interp(t, self.t, self.FPrime[:, 0, 1], left=np.nan,
                         right=np.nan)

    def interp_FPrime22(self, t):
        return np.interp(t, self.t, self.FPrime[:, 1, 1], left=np.nan,
                         right=np.nan)


class ProteanEnsemble():

    strl_ensemble = None

    deformations = None

    dt = None

    def __init__(self, dt, strl_ensemble, alphas=None, Qs=None):
        self.dt = dt
        self.strl_ensemble = strl_ensemble
        self.deformations = []
        if alphas is None:
            alphas = [None for strl in self.strl_ensemble]
            Qs = [None for strl in self.strl_ensemble]
        for strl, al, Q in zip(self.strl_ensemble, alphas, Qs):
            self.deformations.append(ProteanDeformationTensor(strl,
                                                              self.dt,
                                                              alpha=al,
                                                              Q=Q))

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.deformations):
            i = self.i
            self.i += 1
            return self.deformations[i]
        else:
            raise StopIteration

    def __getitem__(self, key):
        return self.deformations[key]

    @property
    def t(self):
        t = np.array([0])
        for d in self.deformations:
            if not d.streamline.failed:
                if d.t[-1] > t[-1]:
                    t = d.t
        return t

    @property
    def s(self):
        s = []
        for d in self.deformations:
            if not d.streamline.failed:
                s.append(d.streamline.cal_s_at_t(d.t))

        length = [len(si) for si in s]
        max_len = max(length)
        s = [np.pad(e, (0, max_len-len(e)),
                    mode="constant",
                    constant_values=np.nan) for e in s]
        return np.array(s)

    @property
    def v(self):
        v = []
        for d in self.deformations:
            if not d.streamline.failed:
                vi = d.streamline.pm.get_v(d.r)
                v.append(np.linalg.norm(vi, axis=1))

        length = [len(vi) for vi in v]
        max_len = max(length)
        v = [np.pad(e, (0, max_len-len(e)),
                    mode="constant",
                    constant_values=np.nan) for e in v]
        return np.array(v)

    @property
    def epsi11(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 0, 0])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @property
    def epsi12(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 0, 1])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @property
    def epsi13(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 0, 2])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @property
    def epsi22(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 1, 1])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @property
    def epsi23(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 1, 2])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @property
    def epsi33(self):
        epsi = []
        for d in self.deformations:
            if not d.streamline.failed:
                epsi.append(d.epsiPrime[:, 2, 2])

        length = [len(ei) for ei in epsi]
        max_len = max(length)
        epsi = [np.pad(e, (0, max_len-len(e)),
                       mode="constant",
                       constant_values=np.nan) for e in epsi]
        return np.array(epsi)

    @staticmethod
    def cal_FPrime(t, absV, epsiPrime):
        """ this function is the same as in the class ProteanDeformationTensor
        but in can integrate a whole ensemble of epsiPrime at the same time
        """
        F11 = absV/absV[:, 0:1]
        intEpsi22 = sc.integrate.cumtrapz(epsiPrime[:, :, 1, 1], x=t, initial=0, axis=1)
        F22 = np.exp(intEpsi22)
        intEpsi33 = sc.integrate.cumtrapz(epsiPrime[:, :, 2, 2], x=t, initial=0, axis=1)
        F33 = np.exp(intEpsi33)
        F12 = absV * sc.integrate.cumtrapz(epsiPrime[:, :, 0, 1]*F22/absV,
                                           t, initial=0, axis=1)
        F23 = F22 * sc.integrate.cumtrapz(epsiPrime[:, :, 1, 2]*F33/F22, x=t,
                                          initial=0, axis=1)
        F13 = absV * sc.integrate.cumtrapz((epsiPrime[:, :, 0, 1]*F23 +
                                            epsiPrime[:, :, 0, 2]*F33)/absV,
                                           x=t, initial=0, axis=1)
        F = np.zeros(epsiPrime.shape)
        F[:, :, 0, 0] = F11
        F[:, :, 0, 1] = F12
        F[:, :, 0, 2] = F13
        F[:, :, 1, 1] = F22
        F[:, :, 1, 2] = F23
        F[:, :, 2, 2] = F33
        return F

    def update_time(self):
        for dfmt in self.deformations:
            dfmt.update_time()

    def cal_protean_rotation(self, cores=2, alpha0=0):
        t0 = time.time()
        if alpha0 == 0:
            alpha0 = np.zeros(len(self.deformations))
        batch = 0
        finish = False
        while not finish:
            nmin = batch*50
            nmax = (batch+1)*50
            if nmax >= len(self.deformations):
                nmax = len(self.deformations)
                finish = True
            res = Parallel(n_jobs=cores)(delayed(dfm.cal_protean_rotation)(a0)
                                         for dfm, a0 in
                                         zip(self.deformations[nmin:nmax],
                                             alpha0))
            for res_i, dfm in zip(res, self.deformations[nmin:nmax]):
                dfm._Q = res_i[0]
                dfm._alpha = res_i[1]
            print(nmax)
            batch += 1

        print(f"time = {time.time()-t0}")

    def return_save_dict(self):
        strl_dict = self.strl_ensemble.return_save_dict()
        deform_dict = {'dt': self.dt}
        deform_dict['alphas'] = [dfmt._alpha for dfmt in self.deformations]
        deform_dict['Q'] = [dfmt.Q for dfmt in self.deformations]
        save_dict = {'streamlines': strl_dict, 'deformations': deform_dict}
        return save_dict

    @staticmethod
    def combine_ensemble(*args):
        stlEns = [ens.strl_ensemble for ens in args]
        MasterStrlEns = LPEnsemble.combine_ensemble(*stlEns)
        dt = args[0].dt
        for ens in args:
            if ens.dt != dt:
                raise ValueError("Ensembles with different dt can not be"
                                 + " combined")
        MasterProtEns = ProteanEnsemble(dt, MasterStrlEns)
        protEns = []
        for ens in args:
            protEns += ens.deformations
        MasterProtEns.deformations = protEns
        return MasterProtEns

    def create_property_string(self):
        date_str = date.today().strftime("%Y%m%d")
        # specs about porous media
        pm_spec = self.strl_ensemble.pm.get_pm_spec()
        pm_str = pm_spec["name"]+str(pm_spec["orientation"])
        # specs about deformation ensemble
        N = len(self.deformations)
        strl = self.strl_ensemble
        strl_dt_max = self.strl_ensemble.dt_max
        return (f"{date_str}_{pm_str}_N{N}_tfi{strl.tfi}_sfi{strl.sfi}_dtMast{strl_dt_max}"
                + f"_dtDef{self.dt}")

    def save(self, path="."):
        filename = f"{self.create_property_string()}.pickle"
        filename = path+"/"+filename
        if os.path.exists(filename):
            # If the file already exists, add a number to the filename
            index = 1
            while True:
                new_filename = f"{os.path.splitext(filename)[0]}_{index}{os.path.splitext(filename)[1]}"
                if not os.path.exists(new_filename):
                    break
                index += 1
            filename = new_filename
        self.filename = filename
        save_dict = self.return_save_dict()
        with open(filename, 'wb') as handle:
            pickle.dump(save_dict, handle)

    def export_epsi(self):
        try:
            source = os.path.split(self.filename)
            path = source[0]
            fn = os.path.splitext(source[1])[0]
        except AttributeError:
            print("you must save before exporting")
        # export epis
        for ij in ['11', '12', '13', '22', '23', '33']:
            description = (' Exported by: export function in ProteanEnsemble \n'
                           + f' Source file: {source} \n'
                           + f' epsilon^Prime_{ij}: {ij} component of velocity gradient tensor in units of 1/sec \n'
                           + ' Rows correspond to different streamlines, columns to different times \n'
                           + ' First row is time in units of sec')
            filename = f"{fn[:8]}_epsi{ij}_{fn[9:]}.txt"
            if ij == '11':
                epsi = self.epsi11
            elif ij == '12':
                epsi = self.epsi12
            elif ij == '13':
                epsi = self.epsi13
            elif ij == '22':
                epsi = self.epsi22
            elif ij == '23':
                epsi = self.epsi23
            elif ij == '33':
                epsi = self.epsi33
            np.savetxt(os.path.join(path, filename),
                       np.vstack((self.t, epsi)),
                       header=description, comments="#")

    def export_s(self):
        try:
            source = os.path.split(self.filename)
            path = source[0]
            fn = os.path.splitext(source[1])[0]
        except AttributeError:
            print("you must save before exporting")
        description = (' Exported by: export function in ProteanEnsemble \n'
                        + f' Source file: {source} \n'
                        + ' advection distance along streamline in units of m \n'
                        + ' Rows correspond to different streamlines, columns to different times \n'
                        + ' First row is time in units of sec')

        fn = f"{fn[:8]}_s_{fn[9:]}.txt"
        np.savetxt(os.path.join(path, fn),
                   np.vstack((self.t, self.s)),
                   header=description, comments="#")

    def export_v(self):
        try:
            source = os.path.split(self.filename)
            path = source[0]
            fn = os.path.splitext(source[1])[0]
        except AttributeError:
            print("you must save before exporting")
        description = (' Exported by: export function in ProteanEnsemble \n'
                        + f' Source file: {source} \n'
                        + ' absolute value of velocity in units of m/s \n'
                        + ' Rows correspond to different streamlines, columns to different times \n'
                        + ' First row is time in units of sec')

        fn = f"{fn[:8]}_v_{fn[9:]}.txt"
        np.savetxt(os.path.join(path, fn),
                   np.vstack((self.t, self.v)),
                   header=description, comments="#")

    def export_units(self, path="."):
        pm = self.strl_ensemble.pm
        flow_orient = pm._orientation
        description = (' Exported by: export function in ProteanEnsemble \n'
                        + f' BCC: {flow_orient} \n'
                        + ' first row: typical advection time tau0=d/<v>_e [sec]\n'
                        + ' second row: bead diameter: d [m]')

        tau0 = pm.cal_tau0()
        d = pm._d
        fn = f"BCC{flow_orient}_units.txt"
        np.savetxt(fn,
                   np.array([[tau0], [d]]),
                   header=description, comments="#")

    @staticmethod
    def load(path, filename, pm, sel=slice(None, None, None)):
        with open(f'{path}/{filename}', 'rb') as handle:
            ens_saved = pickle.load(handle)
        # print(ens_saved['streamlines'].keys())
        strl =  ens_saved['streamlines']
        strl['r'] = strl['r'][sel]
        strl['t'] = strl['t'][sel]
        strl['s'] = strl['s'][sel]
        strl['failed'] = strl['failed'][sel]
        streamlines = LPEnsemble.dict_to_ensemble(ens_saved['streamlines'], pm)
        pe = ProteanEnsemble(ens_saved['deformations']['dt'],
                             streamlines,
                             ens_saved['deformations']['alphas'][sel],
                             ens_saved['deformations']['Q'][sel])
        print("Positions need to be interpolated. This may take a while...")
        _ = pm.generateValidPoint()
        for i, prot in enumerate(pe):
            print(i)
            _ = prot.Q # needed to interpolate positions prot._r
        return pe
