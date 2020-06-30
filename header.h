#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>

#define DAT double

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


class Problem
{
private:
    const DAT Lx      = 0.025;      // domain length, m
    const DAT Ly      = 0.025;
    const DAT K       = 9.87e-12;   // intrinsic permeability, m^2
    const DAT phi     = 0.4;        // porosity
    const DAT muw     = 1.2e-3;     // water dynamic viscoity, Pa*s
    const DAT rhow    = 1e3;        // water density, kg/m^3
    const DAT rho_s   = 1e3;        // solid density, kg/m^3
    const DAT sstor   = 1e-6;       // specific storage coefficient, 1/m
    const DAT g       = 9.81;       // m/s^2
    const DAT vg_a    = 1.5;        // van Genuchten pore parameter
    const DAT vg_n    = 1.35;       // van Genuchten pore parameter
    const DAT vg_m    = 1. - 1./vg_n;
    const DAT E       = 3.5e6;      // Young's modulus, MPa
    const DAT nu      = 0.3;        // Poisson ratio
    const DAT lam     = E*nu/(1+nu)/(1-2*nu);
    const DAT mu      = E/2/(1+nu);


    // Numerics
    const int nx      = 64;       // number of cells
    const int ny      = nx;
    const DAT dx      = Lx/nx;    // cell size
    const DAT dy      = Ly/ny;
    const DAT niter   = 1e5;      // number of PT steps
    const DAT eps_a_m = 1e-9;     // absolute tolerance, mechanics
    const DAT eps_a_h = 1e-6;     // absolute tolerance, flow
    const DAT damp    = 1e1;

    const DAT dt        = 1e-1;    // Seconds
    const DAT Time      = dt*1;
    const DAT nt        = Time / dt;

    bool do_mech   = true;
    bool do_flow   = false;
    bool save_mech = true;
    bool save_flow = true;

    int save_intensity;

    // Unknowns
    DAT *Pw;        // Water pressure
    DAT *Sw;        // Water saturation
    DAT *Pw_old;
    DAT *Sw_old;
    DAT *qx, *qy;   // Fluid fluxes
    DAT *Krx, *Kry; // Relative permeabilities for water
    DAT *rsd_h;
    DAT *Txx, *Tyy; // Stresses
    DAT *Txy;
    DAT *Vx, *Vy;   // Solid velocities
    DAT *Ux, *Uy;   // Solid displacements
    DAT *rsd_m_x;
    DAT *rsd_m_y;
    // Unknowns on GPU
    DAT *dev_Pw;
    DAT *dev_Sw;
    DAT *dev_Pw_old;
    DAT *dev_Sw_old;
    DAT *dev_qx, *dev_qy;
    DAT *dev_Krx, *dev_Kry;
    DAT *dev_rsd_h;
    DAT *dev_Txx, *dev_Tyy;
    DAT *dev_Txy;
    DAT *dev_Vx, *dev_Vy;
    DAT *dev_Ux, *dev_Uy;
    DAT *dev_Ux_old, *dev_Uy_old;
    DAT *dev_rsd_m_x;
    DAT *dev_rsd_m_y;

    std::string respath;

    // Functions calculating unknowns over whole mesh
    void H_Substep_GPU();     // Hydro substep
    void M_Substep_GPU();     // Mechanical substep
    void SetIC_GPU();         // Set initial conditions
    void Compute_Sw_GPU();    // Compute water saturation
    void Compute_Kr_GPU();    // Compute relative permeability
    void Compute_Q_GPU();     // Compute fluid fluxes using Kr
    void Update_Pw_GPU();     // Compute residual and update water pressure
    void Update_V_GPU();      // Update solid
    void Update_U_GPU();      // Update solid displacement (includes damping of V!)
    void Update_Stress_GPU(); // Update stress and compute residual

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
