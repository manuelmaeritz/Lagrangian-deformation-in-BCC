import scipy as sc
import numpy as np
import os
from datetime import date
import pickle
from joblib import Parallel, delayed


class LP():

    pm = None

    r = None

    t = None

    s = None

    dt_max = None

    failed = False

    direction = None
    """ int: plus or minus one for forward or backward advection
    """

    def __init__(self, pm, r, t, s, dt_max, failed=False, direction=1):
        self.dt_max = dt_max
        self.pm = pm
        if len(r) != len(t) or len(s) != len(r):
            raise ValueError("r and t is not of same length")
        self.t = t
        self.r = r
        self.s = s
        self.failed = failed
        self.direction = direction

    def get_v(self, t, r):
        """ This function is not to evaluate velocities at lists of positions!
        The function is mainly supposed to be called by the RK45 advection
        function.
        r: 1d array
        """
        r = r.reshape((1, -1))
        v = self.direction * self.pm.get_v(r)
        return v

    def advectRK(self, tfi=None, sfi=None):
        t0 = self.t[-1]
        r0 = self.r[-1]
        if (tfi is None) and (sfi is None):
            raise Exception("Either tfi, sfi, or T must be passed")
        rk = sc.integrate.RK45(self.get_v, t0, r0, np.inf, max_step=self.dt_max,
                               vectorized=False)
        finish = False
        while not finish:
            v_last = self.get_v(0, self.r[-1])
            if np.any(np.isnan(v_last)):
                print("Particle stepped on nan velocity")
                self.failed = True
                break
            if np.diff(self.t[-2:]) < 1e-7:
                print("Advection step too small")
                self.failed = True
                break
            if rk.status == 'failed':
                print("Trajectory solving failed")
                self.failed = True
                break
            rk.step()
            self.r = np.append(self.r, rk.y.reshape((1, -1)), axis=0)
            self.t = np.append(self.t, rk.t)
            s = self.s[-1] + np.linalg.norm(v_last)*(self.t[-1]-self.t[-2])
            self.s = np.append(self.s, s)
            try:
                tfi_cond = self.t[-1] >= tfi
            except TypeError:
                tfi_cond = True
            try:
                sfi_cond = self.s[-1] >= sfi
            except TypeError:
                sfi_cond = True
            if tfi_cond and sfi_cond:
                finish = True
        return self.r, self.t, self.s, self.failed

    def advectEE(self,):
        raise NotImplementedError

    def cal_t_at_s(self, s):
        """ interpolates the support points of time and advection distance
        self.t, self.s to return the time t at the distance s
        """
        t_supp = np.array(self.t)
        s_supp = np.array(self.s)
        # s = s[s < max(s_supp)]
        v0 = np.linalg.norm(self.get_v(0, self.r[0])[0])
        vend = np.linalg.norm(self.get_v(0, self.r[-1])[0])
        spline = sc.interpolate.CubicSpline(s_supp, t_supp,
                                            bc_type=((1, 1/v0), (1, 1/vend)),
                                            extrapolate=False)
        return spline(s)

    def cal_s_at_t(self, t):
        t_supp = np.array(self.t)
        s_supp = np.array(self.s)
        # s = s[s < max(s_supp)]
        v0 = np.linalg.norm(self.get_v(0, self.r[0])[0])
        vend = np.linalg.norm(self.get_v(0, self.r[-1])[0])
        spline = sc.interpolate.CubicSpline(t_supp, s_supp,
                                            bc_type=((1, v0), (1, vend)),
                                            extrapolate=False)
        return spline(t)

    def _r_interp(self, t):
        try:
            m = np.argmax(self.t[self.t <= t])
        except ValueError:
            return np.array([np.nan, np.nan, np.nan])
        if m+1 >= len(self.t):
            return np.array([np.nan, np.nan, np.nan])
        vm = self.get_v(0, self.r[m])[0]
        vp = self.get_v(0, self.r[m+1])[0]
        spline = sc.interpolate.CubicSpline(self.t[m:m+2], self.r[m:m+2],
                                            bc_type=((1, vm), (1, vp)))
        return spline(t)

    def r_interp(self, t):
        r = []
        for ti in t:
            ri = self._r_interp(ti)
            r.append(ri)
        r = np.array(r)
        return r

    def get_as_dict(self):
        dic = {"r": self.r,
               "t": self.t,
               "pm": self.pm.get_pm_spec(),
               "dt_max": self.dt_max}
        return dic


class LPEnsemble:

    pm = None

    _lpe = None

    dt_max = None

    sfi = None

    tfi = None

    def __init__(self, pm, traj, times, s, dt_max, failed=False, sfi=None,
                 tfi=None):
        self.pm = pm
        _ = self.pm.generateValidPoint()
        self.dt_max = dt_max
        t0 = times[0][0]
        s0 = s[0][0]
        self._lpe = []
        if not failed:
            failed = [False for t in traj]
        for r, t, s, f in zip(traj, times, s, failed):
            if t[0] != t0:
                raise ValueError("All starting times in an ensemble need to "
                                 + "be the same")
            if s[0] != s0:
                raise ValueError("All starting positions in an ensemble need "
                                 + "to be the same")
            try:
                lp = LP(self.pm, r, t, s, self.dt_max, f)
            except ValueError as e:
                raise e
            self._lpe.append(lp)
        self.sfi = sfi
        self.tfi = tfi

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self._lpe):
            i = self.i
            self.i += 1
            return self._lpe[i]
        else:
            raise StopIteration

    def advectRK(self, sfi=None, tfi=None, cores=2):
        self.tfi = tfi
        self.sfi = sfi
        batch = 0
        finish = False
        while not finish:
            nmin = batch*50
            nmax = (batch+1)*50
            if nmax >= len(self._lpe):
                nmax = len(self._lpe)
                finish = True
            result = Parallel(n_jobs=cores)(delayed(lp.advectRK)(sfi=sfi,
                                                                 tfi=tfi) for
                                            lp in self._lpe[nmin:nmax])
            for res, lp in zip(result, self._lpe[nmin:nmax]):
                lp.r = res[0]
                lp.t = res[1]
                lp.s = res[2]
                lp.failed = res[3]
            print(nmax)
            batch += 1

    def return_save_dict(self):
        save_dic = dict()
        save_dic['dt_max'] = self.dt_max
        save_dic['sfi'] = self.sfi
        save_dic['tfi'] = self.tfi
        t = []
        r = []
        s = []
        failed = []
        for lp in self._lpe:
            t.append(lp.t)
            r.append(lp.r)
            s.append(lp.s)
            failed.append(lp.failed)
        save_dic['r'] = r
        save_dic['t'] = t
        save_dic['s'] = s
        save_dic['failed'] = failed
        save_dic['pm'] = self.pm.get_pm_spec()
        return save_dic

    def split_ensemble(self, n):
        dic = self.return_save_dict()
        k, m = divmod(len(self._lpe), n)
        split_r = [dic['r'][i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in
                   range(n)]
        split_t = [dic['t'][i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in
                   range(n)]
        split_s = [dic['s'][i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in
                   range(n)]
        split_f = [dic['failed'][i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in
                   range(n)]
        split_ens = [LPEnsemble(self.pm, r, t, s, dic['dt_max'], f,
                                sfi=dic['sfi'], tfi=dic['tfi'])
                     for r, t, s, f in zip(split_r, split_t, split_s, split_f)]
        return split_ens

    @staticmethod
    def combine_ensemble(*args):
        pm = args[0].pm
        dt_max = args[0].dt_max
        sfi = args[0].sfi
        tfi = args[0].tfi
        for ens in args:
            if ens.dt_max != dt_max:
                raise ValueError("Can not combine ensemble with different"
                                 + " dt_max")
            if ens.pm != pm:
                raise ValueError("Can not combine ensemble with different"
                                 + " porous mediy")
            if (ens.sfi != sfi or ens.tfi != tfi):
                raise ValueError("Can not combine ensemble with different"
                                 + " termination condition")
        masterEnsemble = LPEnsemble(pm, [np.array([[0, 0, 0]])],
                                    [np.array([0])], [np.array([0])], dt_max,
                                    tfi=tfi, sfi=sfi)
        comb_lpe = []
        for lpe in args:
            comb_lpe += lpe._lpe
        masterEnsemble._lpe = comb_lpe
        return masterEnsemble

    def save(self, path):
        date_str = date.today().strftime("%Y%m%d")
        # specs about porous media
        pm_spec = self.pm.get_pm_spec()
        pm_str = pm_spec["name"]+str(pm_spec["orientation"])
        # specs about ensemble
        N = len(self._lpe)
        if self.sfi is None:
            term = f"tfi{self.tfi}"
        else:
            term = f"sfi{self.sfi}"
        strl_dt_max = self.dt_max
        i = 0
        file_ex = True
        while file_ex:
            filename = (f"{date_str}_strlEns_{pm_str}_N{N}_{term}_"
                        + f"dtMax{strl_dt_max}_{i}")
            file_ex = os.path.isfile(path+"/"+filename+".pickle")
            i += 1
        save_dict = self.return_save_dict()
        with open(f'{path}/{filename}.pickle', 'wb') as handle:
            pickle.dump(save_dict, handle)

    @staticmethod
    def load(path, filename, pm):
        with open(f'{path}/{filename}', 'rb') as handle:
            dic = pickle.load(handle)
        lp_ens = LPEnsemble.dict_to_ensemble(dic, pm)
        return lp_ens

    @staticmethod
    def dict_to_ensemble(dic, pm):
        if dic['pm']['v_min'] != pm._v_min:
            raise Exception('v_min of porous medium must be the same as the '
                            + 'one used for the ensemble!')
        lp_ens = LPEnsemble(pm, dic['r'], dic['t'], dic['s'], dic['dt_max'],
                            dic['failed'], dic['sfi'], dic['tfi'])
        return lp_ens
