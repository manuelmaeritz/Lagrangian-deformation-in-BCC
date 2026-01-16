# Lagrangian deformation in BCC-flow
This repository contains the code for lagrangian particle tracking in a laminar flow through an array of beads in a body-center cubic structure as well as the code to compute the fluid deformations in the vicinity of the particles.
The streamlines are computed by numerically solving the ODE

$$\frac{d \mathbf{r}}{dt}=\pmb{u}(\pmb{r})$$

with a Runge-Kutta method of order 4. The fluid deformations are quantified by the deformation gradient tensor $\pmb{F}$ that integrates first order velocity flucutations along the particle trajectories
$$\frac{d \pmb{F}}{dt}=\pmb{\epsilon} \pmb{F}$$,
where $\pmb{\epsilon}=\nabla \pmb{u}$. Here the deformation gradient tensor is computed in the so called Protean coordinates as described in [Lester et al. 2018; JFM]. In this coordinate frame the deformation gradient tensor is an upper triangular matrix.

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
