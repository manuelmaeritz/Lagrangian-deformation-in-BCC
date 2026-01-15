import numpy as np
import pyvista as pv


class Curtain():

    def __init__(self, dx_max):
        self.nodes = None
        self.cells = []
        self.time = None
        self.dx_max = dx_max

    def add_first_line(self, r, clr, time):
        dl = np.linalg.norm(np.diff(r, axis=0), axis=1)
        intv = int(self.dx_max/dl[0])
        idx = np.arange(0, len(r), intv)
        idx = np.append(idx, len(r)-1)
        self.nodes = r[idx]
        self.time = time[idx]
        self.last_line_clr = clr[idx]
        # self.nodes = r[::intv]
        # self.last_line_clr = clr[::intv]
        self.last_line_idx = np.arange(len(self.nodes))

    def select_next_ptcls(self, r, clr, time):
        # if left and right of nan is separated by no more than dx_max, remove
        # nan
        nan = np.where(np.isnan(clr))[0]
        dist = np.linalg.norm(r[nan-1]-r[nan+1], axis=1)
        no_nan = nan[dist < self.dx_max]
        r = np.delete(r, no_nan, axis=0)
        clr = np.delete(clr, no_nan)
        # print(clr[:100])
        # select particles of same color as in last row
        last_clr_sel = np.isin(clr, self.last_line_clr)
        # select nan values
        nan_sel = np.isnan(r[:, 0])
        sel = np.where(nan_sel+last_clr_sel)[0]
        # print(sel)
        # print(clr[sel][:100])
        # if distance between particles is getting to large add particle in
        # middle
        dist = np.linalg.norm(np.diff(r[sel], axis=0), axis=1)
        large = np.where(dist > self.dx_max)[0]
        ref_idx = ((sel[large]+sel[large+1])/2).astype(int)
        ref_idx[np.isnan(clr[ref_idx])] += 1
        sel = np.insert(sel, large+1, ref_idx)
        # print(clr[sel])
        # if there is just one particle between two nans, delete this particle
        nan = np.where(np.isnan(clr[sel]))[0]
        isolated = nan[np.where(np.diff(nan) < 3)[0]]+1
        sel = np.delete(sel, isolated)
        # print(clr[sel])
        # delete double numbers
        dd_nb = np.where(clr[sel[:-1]] == clr[sel[1:]])[0]
        dd_nan = np.where(np.isnan(clr[sel[:-1]])*np.isnan(clr[sel[1:]]))[0]
        dd = np.concatenate((dd_nb, dd_nan))
        sel = np.delete(sel, dd)
        # print(clr[sel])
        return r[sel], clr[sel], time[sel]

    def add_line(self, r, clr, time):
        this_line_idx = np.arange(len(clr))+len(self.nodes)
        self.nodes = np.concatenate((self.nodes, r), axis=0)
        self.time = np.concatenate((self.time, time), axis=0)
        at_end = False
        i = 0 
        while not at_end:
            #  we always start with a color where the parent still exists
            #  because we never refine on borders or next to nan values
            c = clr[i]
            if not np.isnan(c):
                cell = [0]  # zerro will be overwritten later with the number of nodes
                cell.append(this_line_idx[i])
                # go as long to the right until we either hit a nan value
                # or find a particle, the parent of which still exists
                move_right = True
                while move_right:
                    i += 1
                    c_r = clr[i]
                    if not np.isnan(c_r):
                        cell.append(this_line_idx[i])
                    else:
                        # if nan value is encountered go to parent of starting node
                        parent_pos = np.where(self.last_line_clr == c)[0][0]
                        parent_clr = self.last_line_clr[parent_pos]
                        cell.append(self.last_line_idx[parent_pos])
                        move_right = False
                    # if parent exists move up
                    if c_r in self.last_line_clr:
                        parent_pos = np.where(self.last_line_clr == c_r)[0][0]
                        parent_clr = c_r
                        cell.append(self.last_line_idx[parent_pos])
                        move_right = False
                # move as long to the left until color is same as for starting point
                while parent_clr != c:
                    parent_pos -= 1
                    cell.append(self.last_line_idx[parent_pos])
                    parent_clr = self.last_line_clr[parent_pos]
                p = len(cell)-1
                cell[0] = p
                if p > 2:
                    self.cells.append(cell)
            else:
                i += 1
            if len(clr) == i+1:
                at_end = True
        self.last_line_idx = this_line_idx
        self.last_line_clr = clr

    def flatten_cells(self):
        return [x for xs in self.cells for x in xs]

    def export(self, filename, path='.'):
        plume = pv.PolyData(self.nodes, self.flatten_cells())
        plume.point_data['time'] = self.time
        plume.save(f'{path}/{filename}.vtk')
