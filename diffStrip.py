import scipy
from pathlib import Path
from porousMedium import BCCPorousMedium
import gc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.interpolate
import pickle
from datetime import datetime
from copy import deepcopy

class DiffusiveStrip():
    """This is the docstring for the Strip class. Needs to be created. """

    # -----------------------------------------------------------------------
    # Attributes related to the advection steps
    # -----------------------------------------------------------------------

    _pm = None
    """ `PorousMedia`: The porous media the strips are in
    """

    _dt = None
    """ time increments for advection. If unequal `None` (and `_dx==None`) the
    advection will happen with equal time steps for each particle.
    """

    _dx = None
    """ space increments for advection. If unequal `None` (and `_dt==None`) the
    advection will happen with equal spacial steps for each particle.
    """

    _deltaSave = None
    """ parameter determining in what time or space increments the history
    variables shall be extended. In case of spacial advection (`_xr` not None)
    this variable shall has spacial units. In case of temporal advection this
    variable has temporal units.
    """

    _dlmax = None
    """ Maximal distance between two adjacent particles bevore getting refined.
    """

    # -----------------------------------------------------------------------
    # Attributes concerning the current state of the strips
    # -----------------------------------------------------------------------

    _d = None
    """ `1d-array`: Current distance between a pair of partices.
    """

    _act_ptl_filt = None
    """ 1d-array: indices of the particles which did not get stuck at low
    velocity fields.
    """

    _max_ptcl_idx = None
    """ int: current maximum index of particles
    """

    @property
    def _r(self):
        """ 2d-array: current particle positions: 
        0'th axis: contains different particles.
        1'st axis: contains the 3 coordinates of a specific particle

        This property accesses the last entry of the history variable `_r_hist`

        """
        return self._r_hist[-1]

    @property
    def _ptc_idx(self):
        return self._ptcl_idx_hist[-1]

    @property
    def _r_act(self):
        """ 2d-array: Like `_r` but gives only the active particles selected by
        `_act_ptl_filt`.

        """
        return self._r[self._act_ptl_filt]

    @property
    def _logRho(self):
        """ 1d-array: current integrated stretching rate: log(rho(t))

        This property accesses the last entry of the history variable
        `_logRho_hist`.

        """
        return self._logRho_hist[-1]

    @property
    def _t(self):
        """ float or 1d-array: In case of temporal advection: curent time of
        the strip. In case of spacial advection: individual curent time for
        each particle.

        This property accesses the last entry of the history variable `_t_hist`

        """
        return self._t_hist[-1]

    @property
    def _tau(self):
        return self._tau_hist[-1]

    @property
    def _x(self):
        """ float or 1d-array: In case of spacial advection: curent
        longitudinal advection distance of the strip. In case of temporal
        advection: individual curent advectin distance of each particle.

        This property accesses the last entry of the history variable `_x_hist`
        """
        return self._x_hist[-1]


    # -----------------------------------------------------------------------
    # History variables
    # -----------------------------------------------------------------------

    _r_hist = None
    """ List: history of particle positions.
    """

    _ptcl_idx_hist = None
    """ List: particle index
    """

    _logRho_hist = None
    """ `List` of `1d-arrays`: History of integrated stretching rate:
        log(rho(t))
    """

    _meanLogRho_hist = None
    """ List: weighted mean of _logRho_hist.
    """

    _tau_hist = None
    """ 1d-array: History of wrapped time.
    """

    _t_hist = None
    """ `List`: time. In case of temporal advection (`_dx=None`) the times
    correxponding to the history variables. In case of spacial advection
    (`_dt=None`) the individual time history for each particle.
    """

    _x_hist = None
    """ `List`: location. In case of spacial advection (`_dt=None`) the
    locations correxponding to the history variables. In case of temporal
    advection (`_dx=None`) the history of individual positons of the particles
    longitudinal to the mean flow direction.
    """

    def __init__(self):
        self._r_hist = []
        self._logRho_hist = []
        self._meanLogRho_hist = []
        self._tau_hist = []
        self._t_hist = []
        self._x_hist = []
        self._ptcl_idx_hist = []

    # + tested
    # - docstring
    def init_r(self, r):
        """ sets the instance variable `_r` by the argument `r`. By doing so
        also the instance variables `_logRho`, `_act_ptl_filt`, `_t` and `_x`
        are re-setted.

        :r: '2d-array': the list of coordinates to be set. len(r) must be even.
        """
        self._r_hist = [r, r]
        ptcl_idx = np.arange(len(r), dtype='float')
        self._max_ptcl_idx = ptcl_idx[-1]
        self._ptcl_idx_hist = [ptcl_idx, ptcl_idx]
        self._logRho_hist = [np.zeros(len(r)-1), np.zeros(len(r)-1)]
        self._act_ptl_filt = np.full(len(r), True)
        d = self.cal_d(r)
        self.upd_d(d)
        self._tau_hist = [np.zeros(len(r)-1), np.zeros(len(r)-1)]
        self._meanLogRho_hist = [0, 0]
        # spacial advection
        if self._dt is None and self._dx is not None:
            self._x_hist = [0.0, 0.0]
            self._t_hist = [np.full(len(r), 0.0), np.full(len(r), 0.0)]
        # temporal advection
        if self._dt is not None and self._dx is None:
            self._t_hist = [0.0, 0.0]
            self._x_hist = [np.full(len(r), 0.0), np.full(len(r), 0.0)]

    # + func has been adapted
    # + doc not yet
    # - has not been tested
    def advectEE(self, order=2):
        """ Explicit Euler advection. Advect each (active) particle either
        with first or second order explicit Euler advection.

        :order: `int` either 1 or 2 (default). Determines the order of the
        advection sheeme.

        :returns: 1d-array: For spacial advection `self._dt == None` it returns
        the advection time 'dt' for each particle. For temporal advection
        `selt._dx == None` it returns the advection distance longitudinal to
        the mean flow direction 'dx' for each particle.
        """
        v = self._pm.get_v(self._r_act)
        gradV = self._pm.get_Jv(self._r_act)
        if order==2:
            du = np.vstack((np.sum(v*gradV[:,0,:], axis=1),
                            np.sum(v*gradV[:,1,:], axis=1),
                            np.sum(v*gradV[:,2,:], axis=1))).T
        # spacial advection
        if self._dt is None and self._dx is not None:
            # Spare an additional evaluation of the velocity field by not using the
            # _pm.get_vr()
            vR = np.dot(v, self._pm._R)
            rR = self._pm.cal_rR(self._r_act)
            if order == 2:
                duR = np.dot(du, self._pm._R)
                dt = (-vR[:, 0]+
                      np.sqrt(vR[:, 0]**2+2*duR[:, 0]*self._dx))/duR[:, 0]
            else:
                dt = dx/vR[:, 0]
            backflow = (dt<=0)
            if np.any(backflow):
                print(f'{np.sum(backflow)} negative advection -> remove')
            dt[backflow] = np.nan
            # if particle is to far from plane, discard
            x_mean = np.nanmean(rR[:, 0])
            far = (np.abs(rR[:, 0]-x_mean)>self._dx)
            if np.any(far):
                print(f'{np.sum(far)} ptcl were to far from plane -> removed')
            dt[far] = np.nan
            # print('dt')
            # print(dt)
            dt = np.reshape(dt, (-1, 1))
        # if self._dt is None and self._dx is not None:
        #     dx = np.nanmin(rR[:,0])+self._dx-rR[:,0]
        #     dt = dx/vR[:,0]
        #     #print(dt)
        #     dt[dt<=0] = np.nan
        #     dt = np.reshape(dt, (-1, 1))

        # temporal advection
        elif self._dt is not None and self._dx is None:
            dt = np.ones((len(self._r_act), 1))*self._dt
        else:
            print('undefined Advection step')
            return False
        dr = v*dt # first order
        if order==2:
            dr += du/2*dt**2
        r_new = np.full(self._r.shape, np.nan)
        r_new[self._act_ptl_filt] = self._r_act + dr
        # spacial advection
        if self._dt is None and self._dx is not None:
            x_new = self._x + self._dx
            dt_full = np.full(len(self._r), np.nan)
            dt_full[self._act_ptl_filt] = np.reshape(dt, (-1))
            t_new = self._t + dt_full
        # temporal advection
        elif self._dt is not None and self._dx is None:
            t_new = self._t + self._dt
            dx_full = np.full(len(self._r), np.nan)
            dx_full[self._act_ptl_filt] = np.dot(dr, self._pm._R)[:,0]
            x_new = self._x + dx_full
        return r_new, x_new, t_new

    # + updated
    # - docstring
    # + tested
    def cal_d(self, r):
        """Calculates the distance between the pairs of particles.

        :returns: tuple of 1d-arrays: First entry: total distance between the
        pairs of particles; second entry: longitudinal distance; third entry:
            transverse distance.

        """
        diff = np.diff(r, axis=0)
        d = np.linalg.norm(diff, axis=1)
        return d

    # + updated
    # - docstring
    # + tested
    def cal_logRho(self, d):
        """Calculates the stretching the pairs of particles has experienced.
        Notice: This function accesses the `_logRho` and `_d` variables to get the
        old value of them. Therefore call this function before updating them.
        In particular do not call `upd_d` and `refine` in an advection step
        before calling `cal_logRho`.

        :d: current total distance between a pair of particles.

        :returns: Tuple of 1d-arrays: Firs entry: total log stretching; second

        """
        # Compute stretching rate
        gamma = np.log(d/self._d)
        # comute integrated streting rate
        logRho = self._logRho + gamma
        return logRho

    # nothing changed here
    def upd_d(self, d):
        """ Update the variable `_d`

        :d: new value for `_d`

        """
        self._d = d

    def upd_x(self, x, append):
        if append:
            self._x_hist.append(x)
        else:
            self._x_hist[-1] = x

    def upd_tau(self, tau, append):
        if append:
            self._tau_hist.append(tau)
        else:
            self._tau_hist[-1] = tau

    def upd_t(self, t, append):
        if append:
            self._t_hist.append(t)
        else:
            self._t_hist[-1] = t

    def upd_r(self, r, append):
        last_idx = deepcopy(self._ptcl_idx_hist[-1])
        last_idx[np.isnan(r[:, 0])] = np.nan
        if append:
            self._r_hist.append(r)
            self._ptcl_idx_hist.append(last_idx)
        else:
            self._r_hist[-1] = r
            self._ptcl_idx_hist[-1] = last_idx

    def upd_logRho(self, logRho, append):
        if append:
            self._logRho_hist.append(logRho)
        else:
            self._logRho_hist[-1] = logRho

    def upd_MeanLogRho(self, meanLogRho):
        """TODO: Docstring for upd_MeanLogRho.
        :returns: TODO

        """
        self._meanLogRho_hist.append(meanLogRho)

    # + function has been adapted
    # - doc not yet
    # + tested
    def refine(self):
        """ Refines the particle distance. If a pair of particles is father
        apart from each other than the maximal allowed distance determined by
        `self._dlmax` the second particle will be relocated at the center of
        the two particles. The distance variable `_d` will be
        recalculated and updated.

        :d: distances between the pairs of particles acording to which it will
        be assessed which pairs need to be refined.

        """
        id_ref = np.less(self._dlmax, self._d, where=np.isnan(self._d)==False,
                         out=np.full(len(self._d), False))
        id_ref = np.where(id_ref)[0]
        r_ref = (self._r[id_ref]+self._r[id_ref+1])/2
        ptcl_idx_ref = np.arange(self._max_ptcl_idx,
                                 self._max_ptcl_idx+len(r_ref))+1
        # work with the hist variables in this function, since history
        # variables shall not be updated in case `_append_hist`==True
        # refine r
        self._r_hist[-1] = np.insert(self._r, id_ref+1, r_ref, axis=0)
        self._ptcl_idx_hist[-1] = np.insert(self._ptc_idx, id_ref+1,
                                            ptcl_idx_ref, axis=0)
        if len(id_ref) > 0:
            self._max_ptcl_idx = ptcl_idx_ref[-1]
        # refine d, d_lg, d_tr
        self._d[id_ref] *= 0.5
        self._d = np.insert(self._d, id_ref, self._d[id_ref])
        # refine logRho
        self._logRho_hist[-1] = np.insert(self._logRho, id_ref+1,
                                          self._logRho[id_ref])
        # refine tau
        self._tau_hist[-1] = np.insert(self._tau, id_ref+1, self._tau[id_ref])
        # refine act_ptl_filt
        self._act_ptl_filt = np.insert(self._act_ptl_filt, id_ref+1,
                                       np.full(len(id_ref), True))
        # refine x or t
        # temporal advection
        if self._dx is None and self._dt is not None:
            x_ref = (self._x[id_ref]+self._x[id_ref+1])/2
            self._x_hist[-1] = np.insert(self._x, id_ref+1, x_ref)
        # spacial advection
        if self._dt is None and self._dx is not None:
            t_ref = (self._t[id_ref]+self._t[id_ref+1])/2
            self._t_hist[-1] = np.insert(self._t, id_ref+1, t_ref)

    # + updated
    # - docstring
    # + tested
    def del_cons_nan(self):
        """TODO: Docstring for del_cons_nan.

        :arg1: TODO
        :returns: TODO

        """
        bad = np.uint8(~self._act_ptl_filt)
        bad_neigh = bad[1:]+bad[:-1]
        id_del = np.where(bad_neigh == 2)[0]
        self._r_hist[-1] = np.delete(self._r, id_del, axis=0)
        self._ptcl_idx_hist[-1] = np.delete(self._ptc_idx, id_del, axis=0)
        self._d = np.delete(self._d, id_del, axis=0)
        self._logRho_hist[-1] = np.delete(self._logRho, id_del, axis=0)
        self._tau_hist[-1] = np.delete(self._tau, id_del, axis=0)
        self._act_ptl_filt = np.delete(self._act_ptl_filt, id_del, axis=0)
        # Temporal advection
        if self._dt is not None and self._dx is None:
            self._x_hist[-1] = np.delete(self._x, id_del, axis=0)
        # Spacial advection
        elif self._dx is not None and self._dt is None:
            self._t_hist[-1] = np.delete(self._t, id_del, axis=0)

    # + updated
    # - docstring
    # + tested
    def upd_act_ptl_filt(self):
        """ Updates the variable `_act_ptl_filt` by checking for nan values in
        `_r`.

        :returns: `bool` True if there exists at least one pair with two
        active particles, False if not.

        """
        # remove particles as active which lay behind for more than dx
        #if self._dx is not None and self._dt is None:
        #    rR = self._pm.cal_rR(self._r)
        #    slow = np.where(rR[:, 0] < (np.nanmax(rR[:, 0]) - self._dx))[0]
        #    self._r[slow] = np.array([np.nan, np.nan, np.nan])

        # remove particles where coordinates are nan
        nan = np.isnan(self._r[:, 0])
        self._act_ptl_filt = ~nan
        # Check whether there exists at least one strip with more than two
        # active adjacent particles
        good_neigh = ~nan[1:]*~nan[:-1]
        if np.any(good_neigh):
            return True
        else: return False


    def decide_append_hist(self):
        """TODO: Docstring for set_append_hist.
        :returns: TODO

        """
        # Temporal advection
        if self._dt is not None and self._dx is None:
            #if self._t_hist[-1]-self._t_hist[-2] > self._deltaSave:
            if np.diff(self._t_hist)[-1] > self._deltaSave:
                print(np.abs(self._t_hist[-1]%self._deltaSave))
                return True
        # Spacial advection
        elif self._dx is not None and self._dt is None:
            #if self._x_hist[-1]-self._x_hist[-2] > self._deltaSave:
            if np.diff(self._x_hist)[-1] > self._deltaSave:
                return True
        return False

    def cal_w(self, d, rho):
        w = d/rho/np.nansum(d/rho)
        return w

    def cal_tau(self, t, rho):
        """TODO: Docstring for cal_tau.

        :t: TODO
        :rho: TODO
        :returns: TODO

        """
        dt = t-self._t
        if self._dt is None and self._dx is not None:
            dt = (dt[1:]+dt[:-1])/2
        tau = self._tau + dt*rho**2
        return tau

    def cal_MeanLogRho(self, logRho, w):
        """TODO: Docstring for cal_MeanLogRho.
        :returns: TODO

        """
        id_good = ~np.isnan(logRho)
        meanLogRho = np.average(logRho[id_good], weights=w[id_good])
        return meanLogRho

    def advect(self, order, end, saveTraj):
        """ Advect the pairs of particles until either all pairs got stuck in
        low velocity fields or the maximal advection time or the maximal
        advection distance for temporal or spacial advection respectively, has
        been reached.

        :end: parameter determining the maximal advction time or maximal
        advection distance
        :returns: `boolean`: True in case some pairs has been advected to the
        end, False if all pairs got stuck in low velocity fields

        """
        finish = False
        while not finish:
            r, x, t = self.advectEE(order=order)
            d = self.cal_d(r)
            logRho = self.cal_logRho(d)
            rho = np.exp(logRho)
            tau = self.cal_tau(t, rho)
            append = self.decide_append_hist()
            if append and not saveTraj:
                w = self.cal_w(d, rho)
                meanLogRho = self.cal_MeanLogRho(logRho, w)
                # cal varLogRho
                self.upd_meanLogRho(meanLogRho)
            self.upd_d(d)
            self.upd_logRho(logRho, append and saveTraj)
            self.upd_r(r, append and saveTraj)
            self.upd_x(x, append and saveTraj)
            self.upd_t(t, append and saveTraj)
            self.upd_tau(tau, append and saveTraj)
            alive = self.upd_act_ptl_filt()
            self.del_cons_nan()
            self.refine()
            if not alive: return False
            # temporal advection
            if self._dt is not None and self._dx is None:
                if append:
                    print(f't={self._t:.2f}')
                    print(f'# particle {len(self._r)}')
                if self._t>end:
                    finish = True
            # spacial advection
            if self._dx is not None and self._dt is None:
                if append:
                    print('x={:.2}'.format(self._x))
                    print(f'# particle {len(self._r)}')
                if self._x>end:
                    finish = True
        return True

    def save(self, path, fn_apx=''):
        """ Save the parameters of the current instance to a dictionary at the
        given path.

        :path: Path at which the current instance shall be saved.

        """
        dic = {}
        dic['dt'] = self._dt
        dic['dx'] = self._dx
        dic['deltaSave'] = self._deltaSave
        dic['dlmax'] = self._dlmax
        dic['r_hist'] = self._r_hist
        dic['ptcl_idx_hist'] = self._ptcl_idx_hist
        dic['logRho_hist'] = self._logRho_hist
        dic['meanLogRho_hist'] = self._meanLogRho_hist
        dic['t_hist'] = self._t_hist
        dic['x_hist'] = self._x_hist
        dic['tau_hist'] = self._tau_hist
        dic['pm'] = self._pm.get_pm_spec()
        subfolder = dic['pm']['name']
        if self._dt is not None and self._dx is None:
            subfolder += '/adv_in_time/strips'
            dxt = 'dt={}'.format(self._dt)
            xtFi = 'tfi={}'.format(self._t)
        if self._dx is not None and self._dt is None:
            subfolder += '/adv_in_space/strips'
            dxt = 'dx={}'.format(self._dx)
            xfi = self._x/self._pm._d
            xtFi = 'xfi={:.2}'.format(xfi)
        if 'theta' in dic['pm']:
            theta = 'theta={:.2}'.format(self._pm._theta)
            phi = 'phi={:.2}'.format(self._pm._phi)
            #subfolder +='/'+theta+'_'+phi
        #path += subfolder
        if len(self._meanLogRho_hist)==0:
            save = 'save={full}'
        else:
            save = 'save={mean}'
        day = datetime.today().strftime('%Y-%m-%d')
        dlmax = 'dlmax={}'.format(self._dlmax)
        filename = f'{day}_{dxt}_{xtFi}_{dlmax}_{fn_apx}.pkl'
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path+'/'+filename, 'wb') as f:
            pickle.dump(dic, f)
    
    def load(path, filename):
        """ Loads the parameters of a saved instance and assign the values to
        the current instance.

        :path: path to the file
        :filename: filename

        """
        with open(path+'/'+filename, 'rb') as f:
            dic = pickle.load(f)
        diffStrip = DiffusiveStrip()
        if dic['pm']['name'] == 'BCC':
            ort = dic['pm']['orientation']
            vmin = dic['pm']['v_min']
            diffStrip._pm = BCCPorousMedium(None, None, ort, vmin)
            theta = dic['pm']['theta']
            phi = dic['pm']['phi']
            diffStrip._pm._theta = theta
            diffStrip._pm._phi = phi
        elif dic['pm']['name'] == 'Souzy_RBP':
            diffStrip._pm = SouzyPorousMedium()
            diffStrip._pm._v2min = dic['pm']['v2min']
        diffStrip._dt = dic['dt']
        diffStrip._dx = dic['dx']
        diffStrip._deltaSave = dic['deltaSave']
        diffStrip._dlmax = dic['dlmax']
        diffStrip._r_hist = dic['r_hist']
        diffStrip._ptcl_idx_hist = dic['ptcl_idx_hist']
        diffStrip._d = diffStrip.cal_d(diffStrip._r)
        diffStrip.upd_act_ptl_filt()
        diffStrip._logRho_hist = dic['logRho_hist']
        diffStrip._t_hist = dic['t_hist']
        diffStrip._x_hist = dic['x_hist']
        diffStrip._tau_hist = dic['tau_hist']
        return diffStrip




if __name__ == '__main__':
    pass
