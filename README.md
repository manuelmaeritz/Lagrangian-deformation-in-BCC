# Lagrangian deformation in BCC-flow
This repository contains the code for lagrangian particle tracking in a laminar flow through an array of beads in a body-center cubic structure as well as the code to compute the fluid deformations in the vicinity of the particles.
The streamlines are computed by numerically solving the ODE

$$\frac{d \mathbf{r}}{dt}=\mathbf{u}(\mathbf{r})$$

with a Runge-Kutta method of order 4. The fluid deformations are quantified by the deformation gradient tensor $\pmb{F}$ that integrates first order velocity flucutations along the particle trajectories

$$\frac{d \mathbf{F}}{dt}=\mathbf{\epsilon} \mathbf{F}$$

where $\mathbf{\epsilon}=\nabla \mathbf{u}$. Here the deformation gradient tensor is computed in the so called Protean coordinates as described in [Lester et al. 2018; JFM]. In this coordinate frame the deformation gradient tensor is an upper triangular matrix.

## General workflow
Here we describe the general workflow boiled down to the most essential commands. A more detailed explaination of the individual modules is provided further down. A complete working script containing all the below steps can be found in `apps/create_ensemble.py`

1. create the flowfield instance of the class `BCCPorousMedium` in the module `porousMedium.py`
```
bcc = BCCPorousMedium(grainCoord, cylCoord, flw_nb, v_min)
bcc.initialize(v_data)
```

2. perform the lagrangian particle tracking with the class `LPEnsemble` in the module `lpt`
```
lpe = LPEnsemble(bcc, r0, t0, s0, dt_max_advect)
lpe.advectRK(tfi=tfi, sfi=sfi, cores=cores)
```

3. compute the fluid deformations with the class `ProteanEnsemble` in the module `deformationTensor`
```
prt_ens = ProteanEnsemble(dt_deform, lpe)
prt_ens.cal_protean_rotation(cores=cores)
```

## The `porousMedia` module
The module `porousMedia` contains the class `BCCPorousMedium`, which ties together all the functionalities of the flowfield. To instanciate this class one has to pass the following parameters:
- `grains`: the coordinates of the grains in the BCC-unit cell
- `cylinders`: coordinates of cylinders connecting adjacent grains
- `orientation`: identifier number for the orientation of the flow with respect to the lattice symmetries
- `v_min`: threshold of minimal velocity (particles that enter flow zones of velocities smaller this threshold will be discarded)

For the correct format of these arguments see the example code in `apps/create_ensemble.py`

Important methods in this class are:
- `initialize`: interpolates the velocity field. This needs to be called before actually using the flow field in other classes
- `uniform_sampling`: returns array of points uniformely distributed in the unit cell
- `cal_rR`: calculates coordinates of particles in a coordinate frame with first axis alined with the mean flow direction
- `get_v`: returns the velocity at a set of coordinate positions
- `get_vR`: same as `get_v` but returned velocities are in a coordinates of a frame the first axis of which aligns with the mean flow direction

## The `lpt` module
