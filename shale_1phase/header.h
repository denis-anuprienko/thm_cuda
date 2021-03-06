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
    const DAT Lx      = 0.012;        // domain length, m
    const DAT Ly      = 0.012;
    const DAT K0      = 1e-18;        // initial intrinsic permeability, m^2
    const DAT muf     = 1.0e-3;       // fluid dynamic viscoity, Pa*s
    const DAT rhof    = 7.0e2;        // fluid density, kg/m^3
    const DAT rhos    = 2.0e3;        // solid density, kg/m^3
    const DAT g       = 10.0;         // m/s^2
    const DAT c_f     = 1./22e9;//4.5e-4*1e-6; // parameter from van Noort and Yarushina, 1/Pa
    const DAT c_phi   = 9e-3*1e-6;  // parameter from van Noort and Yarushina, 1/Pa
    const DAT Pt      = 43e6;         // confining pressure, Pa
    const DAT P0      = 1e5;          // atmospheric pressure, Pa
    const DAT gamma   = 0.028*1e-6;   // exponent factor for permeability function

    // Numerics
    const int nx      = 1;      // number of cells
    const int ny      = 32;
    const DAT dx      = Lx/nx;    // cell size
    const DAT dy      = Ly/ny;
    const DAT niter   = 2e5;      // number of PT steps
    const DAT eps_a_h = 1e-10;     // absolute tolerance, flow

    const DAT dt        = 220*60/1e2;    // Seconds
    const DAT Time      = dt*100;
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
    char *indp_y;
    char *indp_x;
    std::vector<DAT> P_upstr; // Upstream pressure at all moments
    std::vector<DAT> q_dnstr; // Downstream flux at all moments
    // Unknowns on GPU
    DAT *dev_Pf;
    DAT *dev_Pf_old;
    DAT *dev_qx, *dev_qy;
    DAT *dev_Kx, *dev_Ky;
    DAT *dev_phi;
    DAT *dev_rsd_h;
    char *dev_indp_x;
    char *dev_indp_y;

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
