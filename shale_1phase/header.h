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
    const DAT Lx      = 0.012;      // domain length, m
    const DAT Ly      = 0.012;
    const DAT K0      = 1e-18;      // initial intrinsic permeability, m^2
    const DAT muf     = 1.2e-3;     // fluid dynamic viscoity, Pa*s
    const DAT rhof    = 1e3;        // fluid density, kg/m^3
    const DAT rhos    = 1e3;        // solid density, kg/m^3
    const DAT g       = 9.81;       // m/s^2


    // Numerics
    const int nx      = 256;       // number of cells
    const int ny      = nx;
    const DAT dx      = Lx/nx;    // cell size
    const DAT dy      = Ly/ny;
    const DAT niter   = 1e5;      // number of PT steps
    const DAT eps_a_h = 1e-6;     // absolute tolerance, flow
    const DAT damp    = 1e1;

    const DAT dt        = 1e5;    // Seconds
    const DAT Time      = dt*10;
    const DAT nt        = Time / dt;

    bool do_mech   = false;
    bool do_flow   = true;
    bool save_mech = true;
    bool save_flow = true;

    int save_intensity;

    // Unknowns
    DAT *Pf;        // Fluid pressure
    DAT *Pf_old;
    DAT *qx, *qy;   // Fluid fluxes
    DAT *Kx, *Ky;   // Pressure-dependent permeabilities
    DAT *phi;       // Porosity
    DAT *rsd_h;
    // Unknowns on GPU
    DAT *dev_Pf;
    DAT *dev_Pf_old;
    DAT *dev_qx, *dev_qy;
    DAT *dev_Kx, *dev_Ky;
    DAT *dev_phi;
    DAT *dev_rsd_h;

    std::string respath;

    // Functions calculating unknowns over whole mesh
    void H_Substep_GPU();     // Hydro substep
    void M_Substep_GPU();     // Mechanical substep
    void SetIC_GPU();         // Set initial conditions
    void Compute_K_GPU();     // Compute permeability
    void Compute_Q_GPU();     // Compute fluid fluxes using K
    void Update_Pf_GPU();     // Compute residual and update fluid pressure
    void Update_Poro();       // Update porosity based on new fluid pressure values

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
