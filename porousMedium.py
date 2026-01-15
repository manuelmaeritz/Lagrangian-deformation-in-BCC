import scipy
import numpy as np


class PorousMedium():

    def get_v(self, r):
        raise NotImplementedError

    def get_Jv(self, r):
        raise NotImplementedError


class BCCPorousMedium(PorousMedium):
    """Docstring for PorousMedium. Needs to be created """

    _orientation = None
    """ `int`: Flow orientation number
    """

    _phi = None
    """ `Float`: First angle of mean flow direction
    """

    _theta = None
    """ `Float`: Second angle of mean flow direction
    """

    _grains = None
    """ `2d-array`: List of grain coordinates and radii
        0'th axis: n different grains
        1'st axis: 3 grain coordinates and radius
        -> shape(grains)=(n,4)
    """

    _cylinders = None
    """ `2d-array`: List of cylinder coordinates and radii at the contact
    points of grains
    """

    _v_min = None
    """ `float`: threshold on velocity. Minimal allowed velocity.
    """

    _v = None
    """ `function`: interpolated velocity field
    """

    _S = None
    """ Matrix to go from primitive coordinates to cartesian coordinates
    """

    _invS = None
    """ Matrix to go from cartesian coordinates to primitive coordinates
    """

    @property
    def _d(self):
        return self._grains[0, 3]

    def __init__(self, grains, cylinders, orientation, v_min):
        self._S = np.array([[-1, 1, 1],
                            [1, -1, 1],
                            [1, 1, -1]])/np.sqrt(3)
        self._invS = np.array([[0, 1, 1],
                               [1, 0, 1],
                               [1, 1, 0]])*np.sqrt(3)/2
        self._orientation = orientation
        self._grains = grains
        self._cylinders = cylinders
        self._v_min = v_min

    def initialize(self, vel, interpolate=True):
        """ Initialise the porous medium from a given velocity field, grain,
        coordinates and cylinders.

        :vel: `csv` file of velocity field
        :grains: `txt` file of grains
        :cylinders: `txt` file of cylinders (at the contact points of grains)

        """
        # Mean Velocity vector angle
        v_mean = np.mean(vel[:, [3, 7, 11]], axis=0)
        vm_abs = np.linalg.norm(v_mean)
        self._phi = np.arctan2(v_mean[1], v_mean[0])
        self._theta = np.arcsin(v_mean[2] / vm_abs)
        if interpolate:
            # interpolation of velocity field
            self._v = self.interpol_vel_field(vel)

    def interpol_vel_field(self, F):
        """
        Function to make a callable function which give the interpolation of
        the velocity field on a set of point. Load Regis CSV data of velocities
        """
        print('Making SCIPY interpolator... This may take some time...')
        # vertex coordinates
        X = F[:, :3]
        # velocities
        U = np.hstack((F[:, 3].reshape(-1, 1),
                       F[:, 7].reshape(-1, 1),
                       F[:, 11].reshape(-1, 1),
                       F[:, 4:7], F[:, 8:11], F[:, 12:15]))

        # Add symetrical points on the side of the mesh to be sure to have
        # correct interpolation close to boundaries
        print('Adding periodicity...')
        alpha = 0.001  # Percentage of periodic vertex to add
        xtrans = np.array([]).reshape(0, 3)
        utrans = np.array([]).reshape(0, U.shape[1])
        n = np.matmul(self._invS, X.T).T / self._d
        for i in range(3):
            trans = self._S[:, i] * self._d
            nbound = np.where((n[:, i] > 1-alpha))[0]
            xtrans = np.vstack((xtrans, X[nbound, :] - trans))
            utrans = np.vstack((utrans, U[nbound, :]))
            nbound = np.where((n[:, i] < alpha))[0]
            xtrans = np.vstack((xtrans, X[nbound, :] + trans))
            utrans = np.vstack((utrans, U[nbound, :]))
        X = np.vstack((X, xtrans))
        U = np.vstack((U, utrans))

        # Make velocity interpolant from SCIPY linear interpolator
        print('Interpolate velocities...')
        v_int = scipy.interpolate.LinearNDInterpolator(X, U)
        print('Done !')
        return v_int

    @property
    def e_flow(self):
        return np.array([np.cos(self._theta)*np.cos(self._phi),
                         np.cos(self._theta)*np.sin(self._phi),
                         np.sin(self._theta)])

    @property
    def _R(self):
        """ Calculates the rotation matrix according to the angles given as
        arguments

        :theta: `float`: theta angle in deg
        :phi: `float`: phi angle in deg
        :returns: `touple` rotation matrix and inverse rotation matrix

        """
        phi = self._phi
        theta = self._theta
        R = np.dot(np.array([[np.cos(phi), -np.sin(phi), 0],
                             [np.sin(phi), np.cos(phi), 0],
                             [0, 0, 1]]),
                   np.array([[np.cos(theta), 0, -np.sin(theta)],
                             [0, 1, 0],
                             [np.sin(theta), 0, np.cos(theta)]]))
        return R

    @property
    def _RT(self):
        RT = scipy.linalg.inv(self._R)
        return RT

    def get_pm_spec(self):
        spec = {}
        spec['orientation'] = self._orientation
        spec['name'] = 'BCC'
        spec['theta'] = self._theta
        spec['phi'] = self._phi
        spec['v_min'] = self._v_min
        return spec

    def generateValidPoint(self):
        """ Generates a point in the basic unit cell, the velocity of which is
        greater than the minimal allowed velocity by `_v2min`. In particular
        the coordinates are not within a grain or the contact cylinder of two
        grains.

        :returns: `1d-array`: valid coordinates

        """
        valid = False
        while not valid:
            xini = np.random.rand(3).reshape(1, -1)
            v = self.get_v(xini)[0, :]
            if not np.isnan(v[0]):
                valid = True
        return xini[0]

    def uniform_sampling(self, meshsize):
        # normalized primitive vectors
        t1 = self._S[:, 0]
        t2 = self._S[:, 1]
        t3 = self._S[:, 2]

        r = []
        x = meshsize
        while x < self._d:
            y = meshsize
            while y < self._d:
                z = meshsize
                while z < self._d:
                    ri = x*t1 + y*t2 + z*t3
                    r.append(ri)
                    z += meshsize
                y += meshsize
            x += meshsize
        r = np.array(r)
        v = self.get_v(r)
        r = r[~np.isnan(v)[:, 0]]
        return r

    def cal_v_mean_euler(self):
        """ Calculates the eulerian mean of the velocity
        """
        meshsize = 0.0001
        r_eulr = self.uniform_sampling(meshsize)
        v_eulr = self.get_v(r_eulr)
        abs_v_eulr = np.linalg.norm(v_eulr, axis=1)
        abs_v_eulr = np.sqrt(np.sum(v_eulr**2, axis=1))
        # abs_v_eulr = abs_v_eulr[abs_v_eulr > 1e-8]
        return np.mean(abs_v_eulr)

    def cal_tau0(self):
        """ Calculates the typical transition time tau0=lc/v0, where lc is the
        length scale of correlation and v0=alpha*<v>_e a typical velocity
        (<v_e>: eulerian mean of velocity, alpha: parameter of gamma
        distribution describing the velocity distribution) (see Dentz et al.
        2016 PRF). Here alpha=1, lc=d (d: diameter of beads)
        """
        v_mean_euler = self.cal_v_mean_euler()
        tau = self._d/v_mean_euler
        return tau

    def cal_ta(self):
        r = self.uniform_sampling(1e-5)
        v = self.get_v(r)
        mean_v = np.nanmean(v, axis=0)
        abs_mean_v = np.linalg.norm(mean_v)
        ta = self._d/abs_mean_v
        return ta

    def cal_rR(self, r):
        """ calculates particle coordinates in a coordinates system the first
        axis of which is aligned in mean flow direction.

        :r: 2d-array: particle coordinates: 0'th axis: different particles;
        1'st axis: coordinates of the specific particles.
        :Returns: 2d-array: rotated coordinates.
        """
        return np.dot(r, self._R)

    def cal_rMod(self, r):
        """ Calculates the projection of coordinates in the basic BCC unitcell.

        :r: particle coordinates in original representation.
        :Returns: 2d-array: Projection of `r` in basic BCC unitcell
        """
        # rPrime is r in coordinates of primitive basis vectors
        rPrime = np.matmul(self._invS, r.T)
        rPrimeMod = rPrime % self._d
        rMod = np.matmul(self._S, rPrimeMod).T
        return rMod

    def cal_Jv_manually(self, r, dx):
        r = np.array(r)
        dx = np.zeros(r.shape)
        dy = np.zeros(r.shape)
        dz = np.zeros(r.shape)
        dx[:, 0] = dx
        dy[:, 1] = dx
        dz[:, 2] = dx

        r_p_dx = self.cal_rMod(r+dx)
        r_m_dx = self.cal_rMod(r-dx)
        r_p_dy = self.cal_rMod(r+dy)
        r_m_dy = self.cal_rMod(r-dy)
        r_p_dz = self.cal_rMod(r+dz)
        r_m_dz = self.cal_rMod(r-dz)

        dvdx = (self._v(r_p_dx)[:, :3]-self._v(r_m_dx)[:, :3])/2/self._dx
        dvdy = (self._v(r_p_dy)[:, :3]-self._v(r_m_dy)[:, :3])/2/self._dx
        dvdz = (self._v(r_p_dz)[:, :3]-self._v(r_m_dz)[:, :3])/2/self._dx

        Jv = np.stack((dvdx, dvdy, dvdz), axis=1)
        Jv = np.transpose(Jv, axes=(0, 2, 1))
        return Jv

    def get_v(self, r):
        """ Gives the velocity at each coordinate in `r`

        :r: 2d-array: particle coordinates: 0'th axis: different particles;
        1'st axis: coordinates of the specific particles.
        :Returns: (2d-array): velocity for each point in r
        """
        nan = np.isnan(r[:, 0])
        v = np.nan*np.ones((len(nan), 3))
        rMod = self.cal_rMod(r[~nan])
        v_good = self._v(rMod)[:, :3]
        v_abs = np.linalg.norm(v_good, axis=1)
        bad = v_abs < self._v_min
        v_good[bad] = np.nan*np.ones((np.sum(bad), 3))
        v[~nan] = v_good
        return v

    def get_Jv(self, r):
        """ Gives the velocity Jacobean at each coordinate in `r`

        :r: 2d-array: particle coordinates: 0'th axis: different particles;
        1'st axis: coordinates of the specific particles.
        :Returns: (3d-array): velocity gradient for each point in r.
        """
        rMod = self.cal_rMod(r)
        Jv = self._v(rMod)[:, 3:]
        return np.reshape(Jv, (-1, 3, 3))

    def get_vR(self, r):
        """ Gives the velocity in rotated coordinate system with first axis
        aligned to the mean flow direction.

        :r: 2d-array: particle coordinates: 0'th axis: different particles;
        1'st axis: coordinates of the specific particles.  :Returns: Tuple:
            First entry is the velocity for each particle (2d-array), second
            entry is the velocity gradient (3d-array).
        """
        v = self.get_v(r)
        vR = np.dot(v, self._R)
        return vR

    def get_RTJvR(self, r):
        """ Gives the velocity Jacobean in rotated coordinate system with first
        axis aligned to the mean flow direction.

        :r: 2d-array: particle coordinates: 0'th axis: different particles;
        1'st axis: coordinates of the specific particles.  :Returns: Tuple:
            First entry is the velocity for each particle (2d-array), second
            entry is the velocity gradient (3d-array).  """
        Jv = self.get_Jv(r)
        RTJvR = np.matmul(np.matmul(self._RT, Jv), self._R)
        return RTJvR

    def export_units(self, path="."):
        flow_orient = self._orientation
        description = (' Exported by: function export_units in porousMedium \n'
                        + f' BCC: {flow_orient} \n'
                        + ' first row: typical advection time tau0=d/<v>_e [sec]\n'
                        + ' second row: bead diameter: d [m]')

        ta = self.cal_ta()
        d = self._d
        fn = f"BCC{flow_orient}_units.txt"
        np.savetxt(fn,
                   np.array([[ta], [d]]),
                   header=description, comments="#")
