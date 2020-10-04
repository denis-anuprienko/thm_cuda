#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>
#include <vector>

#define DAT double

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


class Problem
{
private:
    const DAT Lx      = 0.012;        // Domain length, m
    const DAT Ly      = 0.012;
    const DAT K0      = 1e-18;        // Initial intrinsic permeability, m^2
    const DAT mul     = 1.e-3;       // Liquid dynamic viscoity, Pa*s
    const DAT mug     = 1.e-3;       // Gas    dynamic viscoity, Pa*s
    const DAT rhol    = 7.0e2;        // Liquid density, kg/m^3
    const DAT rhog0   = 7.0e2;        // Gas    density, kg/m^3
    const DAT rhos    = 2.0e3;        // Solid density, kg/m^3
    const DAT g       = 9.81;         // m/s^2
    const DAT c_f     = 1./22e9;//4.5e-4*1e-6; // parameter from van Noort and Yarushina, 1/Pa
    const DAT c_phi   = 9e-3*1e-6;    // parameter from van Noort and Yarushina, 1/Pa
    const DAT Pt      = 43e6;         // Confining pressure, Pa
    const DAT P0      = 1e5;          // Atmospheric pressure, Pa
    const DAT gamma   = 0.028*1e-6;   // Exponent factor for permeability function

    const DAT vg_a    = 0.37;         // van Genuchten pore parameter
    const DAT vg_n    = 2.5;          // van Genuchten pore parameter
    const DAT vg_m    = 1. - 1./vg_n;


    // Numerics
    const int nx      = 64;          // number of cells
    const int ny      = nx;
    const DAT dx      = Lx/nx;        // cell size
    const DAT dy      = Ly/ny;
    const DAT niter   = 2e4;        // number of PT steps
    const DAT eps_a_h = 1e-10;         // absolute tolerance, flow
    const DAT eps_r_h = 1e-5;

    const DAT dt        = 1e3;//220*60/10;    // Seconds
    const DAT Time      = dt*1;
    const DAT nt        = Time / dt;

    bool do_mech   = false;
    bool do_flow   = true;
    bool save_mech = true;
    bool save_flow = true;

    int save_intensity;

    // Unknowns
    DAT *Pl;          // Liquid pressure
    DAT *Pg;          // Gas    pressure
    DAT *Sl;          // Liquid saturation, which also defines gas saturation as 1-Sl
    DAT *Pl_old;
    DAT *Pg_old;
    DAT *Sg_old;
    DAT *Krlx, *Krly; // Liquid relative permeabilities
    DAT *Krgx, *Krgy; // Gas    relative permeabilities
    DAT *qlx, *qly;   // Liquid fluxes
    DAT *qgx, *qgy;   // Gas    fluxes
    DAT *Kx, *Ky;     // Pressure-dependent intrinsic permeabilities
    DAT *phi;         // Porosity
    DAT *rhog;        // Gas density
    DAT *rsd_l;       // Residual of equation for liquid
    DAT *rsd_g;       // Residual of equation for gas
    DAT mass_l;       // Liquid mass (for conservation checking)

    // Unknowns on GPU
    DAT *dev_Pl;
    DAT *dev_Pg;
    DAT *dev_Pc;
    DAT *dev_Sl;
    DAT *dev_Pl_old;
    DAT *dev_Pg_old;
    DAT *dev_Sl_old;
    DAT *dev_Krlx, *dev_Krly;
    DAT *dev_Krgx, *dev_Krgy;
    DAT *dev_qlx, *dev_qly;
    DAT *dev_qgx, *dev_qgy;
    DAT *dev_Kx, *dev_Ky;
    DAT *dev_phi;
    DAT *dev_phi_old;
    DAT *dev_rhog;
    DAT *dev_rsd_l;
    DAT *dev_rsd_g;

    std::string respath;
    std::ofstream flog;

    // Functions calculating unknowns over whole mesh
    void H_Substep_GPU();          // Hydro substep
    void M_Substep_GPU();          // Mechanical substep
    void SetIC_GPU();              // Set initial conditions
    void Compute_K_GPU();          // Compute intrinsic permeability
    void Compute_Kr_GPU();         // Compute relative  permeability
    void Compute_S_GPU();          // Compute liquid saturation (no calculation for gas needed)
    void Compute_Q_GPU();          // Compute fluid fluxes using K
    void Update_P_GPU();           // Update pressure and compute residuals
    void Update_P_impl_GPU();      // "Implicitly" update pressure and compute residuals
    void Update_P_Poro_impl_GPU(); // "Implicitly" update pressure, porosity and compute residuals
    void Update_Poro_GPU();        // Update porosity based on new fluid pressure values
    void Count_Mass_GPU();         // Count mass (of liquid) to check mass conservation

public:
    Problem(){ Init(); }
    //Problem(DAT T_, int nx_, int ny_) : Time(T_), nx(nx_), ny(ny_){}
    ~Problem(){}
    void Init(void);
    void SolveOnGPU(void);
    void SolveOnCPU(void);
    void SaveVTK(std::string path);
    void SaveVTK_GPU(std::string path);
    void SaveDAT_GPU(int stepnum);
};

#endif // HEADER_H
