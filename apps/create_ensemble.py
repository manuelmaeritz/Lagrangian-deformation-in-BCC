from lpt import LPEnsemble
from deformTensor import ProteanEnsemble
from porousMedium import BCCPorousMedium
import time
import gc
import numpy as np


t_start = time.time()
# ---------------------
# Define all parameters
# ---------------------
FLOW_NB = 43
v_min = 1e-9
meshsize = 0.00013
dt_max_advect = 80
sfi = 20*0.002
tfi = 100*3000
cores = 20
dt_deform = 80

# -----------------------
# Load the velocity field
# -----------------------
print("Create porous medium")
FLOWFIELD_DIR = "../../BCC/flow_field/velocity_field/ALL/"
FLOWFIELD = f"BCC_vel_grad_0{FLOW_NB}.csv"
GRAIN_DIR = "../../BCC/flow_field/Coord_grains/"
GRAINS = "BCC_grain_coord_diam_2mm.txt"
CYL = "cylinders_BCC_grain_coord_diam_2mm_ansys.txt"

v_data = np.loadtxt(FLOWFIELD_DIR + FLOWFIELD, skiprows=6, delimiter=',')
grainCoord = np.loadtxt(GRAIN_DIR + GRAINS)
grainCoord = grainCoord*1e-3
cylCoord = np.loadtxt(GRAIN_DIR + CYL, delimiter='\t')
cylCoord *= 1e-3

bcc = BCCPorousMedium(grainCoord, cylCoord, FLOW_NB, v_min)
bcc.initialize(v_data)
del v_data
gc.collect()

# ----------------------------------
# create Lagrangian Particle Ensemble
# ----------------------------------
print("Advect Lagrangian particles")
r0 = bcc.uniform_sampling(meshsize)
r0 = [r.reshape((1, -1)) for r in r0]
t0 = [np.array([0]) for i in r0]
s0 = [np.array([0]) for i in r0]
print(f"Number of particles: {len(r0)}")
lpe = LPEnsemble(bcc, r0, t0, s0, dt_max_advect)
lpe.advectRK(tfi=tfi, sfi=sfi, cores=cores)
split = lpe.split_ensemble(5)
for ens in split:
    ens.save(path="/home/mmaeritz/sim_out")

# ----------------------
# create protean ensemble
# ----------------------
# path = "/home/manuel"
# filenames = ["20240403_strlEns_BCC45_N231_sfi0.04_dtMax80_0.pickle",
#              "20240403_strlEns_BCC45_N231_sfi0.04_dtMax80_1.pickle",
#              "20240403_strlEns_BCC45_N232_sfi0.04_dtMax80_0.pickle",
#              "20240403_strlEns_BCC45_N232_sfi0.04_dtMax80_1.pickle",
#              "20240403_strlEns_BCC45_N232_sfi0.04_dtMax80_2.pickle"]
# for f in filenames:
#     lpe = LPEnsemble.load(path, f, bcc)
# 
#     print("Calculate deformation tensor")
#     prt_ens = ProteanEnsemble(dt_deform, lpe)
#     prt_ens.cal_protean_rotation(cores=cores)
#     prt_ens.save()

print("Calculate deformation tensor")
prt_ens = ProteanEnsemble(dt_deform, lpe)
prt_ens.cal_protean_rotation(cores=cores)
prt_ens.save()

print(f"Total time: {time.time()-t_start}")
