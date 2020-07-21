# thm_cuda
This repository contains code for number of coupled THM problems.
The problems are solved on structured grids with rectangular cells.
The solution method is pseudo-transient method with explicit time stepping, this approach is matrix-free and is suitable for GPU computations with CUDA.

1. The main folder contains code for HM-coupling for bentonite, this includes Richards equation for unsaturated flow and mechanical equilibrium equation with additional terms for bentonite swelling
2. The folder `shale_1phase` contains code for one-phase fluid (CO2) flow in shale with pressure-dependent porosity and permeability
3. The folder `shale_2phase` contains code for two-phase fluid (liquid CO2 and air) flow in shale with pressure-dependent porosity and permeability