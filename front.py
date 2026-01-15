import numpy as np
import pickle
from datetime import datetime
from porousMedium import BCCPorousMedium
from copy import deepcopy
import pyvista as pv
import os
import json


class Front:

    pm = None

    dt = None

    lmax = None

    lmin = None

    dt_save = None

    rim = None
    ''' list of bool: Defines the outer (1d) edge of the structure. Edges in
    the rim will not be refined. Rim edges are defined by True values, interior
    edges by False.
    '''

    t_hist = None

    nodes_hist = None

    edges_hist = None

    direct_export = None

    export_folder = None

    def __init__(self, pm, dt, lmax, lmin, dt_save, direct_export=True,
                 export_folder='./vtk/export'):
        self.pm = pm
        self.dt = dt
        self.lmax = lmax
        self.lmin = lmin
        self.dt_save = dt_save
        self.t_hist = []
        self.nodes_hist = []
        self.edges_hist = []
        self.direct_export = direct_export
        if direct_export:
            if os.path.exists(export_folder):
                export_folder += '_1'
            while os.path.exists(export_folder):
                nb = int(export_folder[-1])
                nb += 1
                export_folder = export_folder[:-1] + f"{nb}"
            self.export_folder = export_folder
            os.makedirs(export_folder)
            self.export_params(export_folder)

    def export_params(self, folder):
        params = {'porous medium': self.pm.get_pm_spec(),
                  'dt': self.dt,
                  'lmax': self.lmax,
                  'lmin': self.lmin,
                  'dt_save': self.dt_save}
        with open(folder + '/advect_param.json', 'w') as f:
            json.dump(params, f)

    def init_front(self, nodes, edges, rim=None):
        self.nodes_hist = [nodes]
        self.edges_hist = [edges]
        self.t_hist = [0]
        if rim is None:
            self.rim = np.full(len(self.edges), False)
        else:
            self.rim = rim

    def init_front_from_pvPolyData(self, pd):
        tri = pd.faces.reshape((-1, 4))
        edges = np.concatenate( (tri[:, [1, 2]], tri[:, [1, 3]], tri[:, [2, 3]]), axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        nodes = np.array(pd.points)
        self.init_front(nodes, edges)
        # for automatic rim detection code not complete yet
        # if rim:
        #     feature_edges = pd.extract_feature_edges(
        #         boundary_edges=True,  # nur Ränder
        #         feature_edges=False,  # keine scharfen Kanten
        #         manifold_edges=False,
        #         non_manifold_edges=False,
        #         )
        #     points = [tuple(row) for row in pd.points]
        #     edge_points = set(tuple(row) for row in feature_edges.points)
        #     rimPoint = np.array([p in edge_points for p in points])

    @property
    def t(self):
        return self.t_hist[-1]

    @t.setter
    def t(self, new_t):
        self.t_hist[-1] = new_t

    @property
    def nodes(self):
        return self.nodes_hist[-1]

    @nodes.setter
    def nodes(self, new_nodes):
        self.nodes_hist[-1] = new_nodes

    @property
    def edges(self):
        return self.edges_hist[-1]

    @edges.setter
    def edges(self, new_edges):
        self.edges_hist[-1] = new_edges

    def cal_l(self, ltest=None):
        l = np.linalg.norm((self.nodes[self.edges[:, 0]] -
                            self.nodes[self.edges[:, 1]]), axis=1)
        if ltest is not None:
            l = ltest
        return l

    def refine(self, ltest=None):
        l = self.cal_l(ltest)
        ref_id = np.where(l > self.lmax)[0]
        # print(ref_id)
        left = self.edges[ref_id, 0]
        right = self.edges[ref_id, 1]
        new_nodes = (self.nodes[left] + self.nodes[right])/2

        new_nodes_id = len(self.nodes) + np.arange(len(new_nodes))
        for l, r, n, rid in zip(left, right, new_nodes_id, ref_id):
            rim = self.rim[rid]
            common_neigh = self.find_common_neighbors(l, r)
            for c in common_neigh:
                self.edges = np.concatenate((self.edges,
                                             np.array([[n, c]], dtype='u4')), axis=0)
            self.rim = np.concatenate((self.rim, np.full(len(common_neigh),
                                                         False)), axis=0)
            self.edges = np.concatenate((self.edges,
                                         np.array([[l, n],
                                                   [n, r]], dtype='u4')),
                                        axis=0)
            self.rim = np.concatenate((self.rim, np.full(2, rim)), axis=0)
            self.edges = np.delete(self.edges, rid, axis=0)
            self.rim = np.delete(self.rim, rid, axis=0)
            ref_id -= 1

        # add new nodes
        self.nodes = np.concatenate((self.nodes, new_nodes), axis=0)
        return len(ref_id)

    def find_neighbors(self, idx, t=-1):
        row, col = np.where(self.edges_hist[t]==idx)
        return self.edges_hist[t][row, 1-col]

    def find_common_neighbors(self, idx1, idx2, t=-1):
        l_neigh = self.find_neighbors(idx1, t=t)
        r_neigh = self.find_neighbors(idx2, t=t)
        common_neigh = np.intersect1d(l_neigh, r_neigh)
        return common_neigh

    def find_triangles(self, t):
        tri = []
        for e in self.edges_hist[t]:
            cn = self.find_common_neighbors(e[0], e[1], t=t)
            for c in cn:
                tri.append([c, e[0], e[1]])
        tri = np.array(tri, dtype='u4')
        tri = np.sort(tri, axis=1)
        tri = np.unique(tri, axis=0)
        tri = np.concatenate((3*np.ones((len(tri), 1), dtype='u4'), tri),
                             axis=1)
        return tri

    def find_edge(self, left, right):
        l = (self.edges[:, 0] == left) + (self.edges[:, 0] == right)
        r = (self.edges[:, 1] == right) + (self.edges[:, 1] == left)
        return l*r

    def get_coarsen_edges(self, ltest=None):
        l = self.cal_l(ltest)
        # coarsen edges which are too long and not on the rim
        crs = (l < self.lmin) * ~self.rim
        crs_edges = self.edges[crs]
        # if one of the nodes of the selected edges is on the rim it must not
        # be coarsened
        rim_nodes = np.unique(self.edges[self.rim].flatten())
        rim_nodes = ~np.isin(crs_edges, rim_nodes)
        crs_edges = crs_edges[rim_nodes[:, 0]*rim_nodes[:, 1]]
        # per node only one edge can be coarsened each time
        # what elements appear twice
        double = np.sort(crs_edges, axis=None)
        double = double[:-1][double[1:] == double[:-1]]
        # remove all but one edge per node
        for d in double:
            row, col = np.where(crs_edges == d)
            crs_edges = np.delete(crs_edges, row[1:], axis=0)
        return crs_edges

    def coarsen(self, ltest=None):
        crs_edges = self.get_coarsen_edges(ltest)

        for id1, id2 in crs_edges:
            new_node = (self.nodes[id1] + self.nodes[id2])/2
            self.nodes[id1] = new_node

            self.edges[self.edges == id2] = id1

        # delete redundant edges
        self.edges = np.sort(self.edges, axis=1)
        self.edges, idx = np.unique(self.edges, return_index=True, axis=0)
        self.rim = self.rim[idx]

        # delete looping edges
        loop = ~(self.edges[:, 0] == self.edges[:, 1])
        self.edges = self.edges[loop]
        self.rim = self.rim[loop]

        # delete right node
        for r in crs_edges[:, 1]:
            self.nodes = np.delete(self.nodes, int(r), axis=0)
            self.edges[self.edges >= r] -= 1
            crs_edges[crs_edges >= r] -= 1
        return len(crs_edges)

    def advectEE(self):
        gradV = self.pm.get_Jv(self.nodes)
        v = self.pm.get_v(self.nodes)
        du = np.vstack((np.sum(v*gradV[:, 0, :], axis=1),
                        np.sum(v*gradV[:, 1, :], axis=1),
                        np.sum(v*gradV[:, 2, :], axis=1))).T
        dr = v*self.dt  # first order
        dr += du/2*self.dt**2
        r_new = self.nodes + dr
        t_new = self.t + self.dt
        return r_new, t_new

    def delete_nan_ptcls(self):
        nan_nodes = np.where(np.isnan(self.nodes[:, 0]))[0]
        for nan in np.flip(nan_nodes):
            # delete connecting edges
            nan_edges, _ = np.where(self.edges == nan)
            self.edges = np.delete(self.edges, nan_edges, axis=0)
            # delete nan particle
            self.nodes = np.delete(self.nodes, nan, axis=0)
            # update edge indices
            self.edges[self.edges > nan] -= 1
        return len(nan_nodes)

    def export(self, t, folder='.'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        nds = self.nodes_hist[t]
        tri = self.find_triangles(t)
        msh = pv.PolyData(nds, tri)
        t = self.t_hist[t]
        msh.save(folder+f'/t={t}.vtk')

    def advect(self, tend):
        if self.direct_export:
            self.export(-1, folder=self.export_folder)
        finish = False
        print('-------------------advect')
        while not finish:
            nodes_new, t_new = self.advectEE()
            if np.abs(self.t%self.dt_save) < self.dt/2:
                print(f't = {self.t}')
                print(f'# particles = {self.nodes.shape}')
                if self.direct_export:
                    print('export')
                    self.nodes = nodes_new
                    self.t = t_new
                    self.export(-1, folder=self.export_folder)
                else:
                    self.t_hist.append(t_new)
                    self.nodes_hist.append(nodes_new)
                    self.edges_hist.append(deepcopy(self.edges_hist[-1]))
            else:
                self.nodes = nodes_new
                self.t = t_new

            ndel = self.delete_nan_ptcls()
            # if ndel > 0:
            #     print(f'{ndel} nan particles deleted')

            nrfn = self.refine()

            ncrs = self.coarsen()
            # if ncrs > 0:
            #     print(f'# particles deleted: {ncrs}')

            if self.t > tend:
                finish = True
        return True

    def save(self, filename, path='.'):
        dic = dict()
        dic['dt'] = self.dt
        dic['lmax'] = self.lmax
        dic['lmin'] = self.lmin
        dic['dtsave'] = self.dt_save
        dic['t_hist'] = self.t_hist
        dic['nodes_hist'] = self.nodes_hist
        dic['edges_hist'] = self.edges_hist
        dic['pm'] = self.pm.get_pm_spec()
        day = datetime.today().strftime('%Y%m%d')
        filename = day+'_'+filename
        with open(f'{path}/{filename}.pkl', 'wb') as f:
            pickle.dump(dic, f)

    def load(filename, path):
        with open(path+'/'+filename, 'rb') as f:
            dic = pickle.load(f)
        if dic['pm']['name'] == 'BCC':
            ort = dic['pm']['orientation']
            vmin = dic['pm']['v_min']
            pm = BCCPorousMedium(None, None, ort, vmin)
            theta = dic['pm']['theta']
            phi = dic['pm']['phi']
            pm._theta = theta
            pm._phi = phi
        dt = dic['dt']
        lmax = dic['lmax']
        lmin = dic['lmin']
        dtsave = dic['dtsave']
        t_hist = dic['t_hist']
        nodes_hist = dic['nodes_hist']
        edges_hist = dic['edges_hist']
        front = Front(pm, dt, lmax, lmin, dtsave)
        front.t_hist = t_hist
        front.nodes_hist = nodes_hist
        front.edges_hist = edges_hist
        return front
