#include "header.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define BLOCK_DIM 16
#define dt_m      1e-6

// Flux is 9.4e g/h = 9.4e-3 kg/h = 9.4e-6/60/60 kg/s

#define FLUX 0//9.4e-3/60/60/0.012//0.012  //9.4e-3/60/60/700 * 10 // kg/m^2/s

void FindMax(DAT *dev_arr, DAT *max, int size);

__global__ void kernel_SetIC(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Sl,
                             DAT *Kx, DAT *Ky,
                             DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                             DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                             DAT *phi,
                             DAT *rsd_l, DAT *rsd_g,
                             const DAT K0,
                             const int nx, const int ny, const DAT Lx, const DAT Ly);


__global__ void kernel_Compute_K(DAT *Pl,
                                 DAT *Kx, DAT *Ky,
                                 const DAT K0, const DAT gamma,
                                 const DAT Pt, const DAT P0,
                                 const int nx, const int ny);

__global__ void kernel_Compute_S(DAT *Pl, DAT *Pg, DAT *Sl,
                                 const DAT rhol, const DAT g,
                                 const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                 const int nx, const int ny);

__global__ void kernel_Compute_Kr(DAT *Sl,
                                  DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                  const DAT vg_m,
                                  const int nx, const int ny);

__global__ void kernel_Compute_Q(DAT *Pl, DAT *Pg, DAT *Sl,
                                 DAT *Kx, DAT *Ky,
                                 DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                 DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                 const DAT rhol, const DAT rhog,
                                 const DAT mul, const DAT mug,
                                 const DAT g,
                                 const int nx, const int ny,
                                 const DAT dx, const DAT dy);

__global__ void kernel_Update_P(DAT *Pl, DAT *Pg,
                                DAT *Sl, DAT *Sl_old,
                                DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                DAT *phi, double *phi_old,
                                DAT *rsd_l, DAT *rsd_g,
                                const DAT rhol, const DAT rhog,
                                const int nx, const int ny,
                                const DAT dx, const DAT dy, const DAT dt);

__global__ void kernel_Update_P_impl(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Pl_old, DAT *Pg_old,
                                     DAT *Sl, DAT *Sl_old,
                                     DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                     DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                     DAT *phi, DAT *phi_old,
                                     DAT *rsd_l, DAT *rsd_g,
                                     const DAT mul, const DAT mug,
                                     const DAT rhol, const DAT rhog,
                                     const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                     const int nx, const int ny,
                                     const DAT dx, const DAT dy, const DAT dt);

// Auxiliary kernel used for debugging on linear diffusion problem
__global__ void kernel_Update_P_diff(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Pl_old, DAT *Pg_old,
                                     DAT *Sl, DAT *Sl_old,
                                     DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                     DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                     DAT *phi, DAT *phi_old,
                                     DAT *rsd_l, DAT *rsd_g,
                                     const DAT mul, const DAT mug,
                                     const DAT rhol, const DAT rhog,
                                     const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                     const int nx, const int ny,
                                     const DAT dx, const DAT dy, const DAT dt);

__global__ void kernel_Update_Poro(DAT *Pl, DAT *Pg,
                                   DAT *Pl_old, DAT *Pg_old,
                                   DAT *Sl, DAT *Sl_old,
                                   DAT *phi,
                                   const DAT c_phi,
                                   const int nx, const int ny);

void FindMax(DAT *dev_arr, DAT *max, int size)
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    cublasCreate(&handle);

    int maxind = 0;
    stat = cublasIdamax(handle, size, dev_arr, 1, &maxind);
    //stat = cublasIsamax(handle, size, dev_arr, 1, &maxind);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Max failed\n");


    cudaMemcpy(max, dev_arr+maxind-1, sizeof(DAT), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
}

void Problem::SetIC_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    printf("Launching %dx%d blocks of %dx%d threads\n", (nx+1+dimBlock.x-1)/dimBlock.x,
           (ny+1+dimBlock.y-1)/dimBlock.y, BLOCK_DIM, BLOCK_DIM);
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg, dev_Pc, dev_Sl,
                                       dev_Kx, dev_Ky,
                                       dev_Krlx, dev_Krly, dev_Krgx, dev_Krgy,
                                       dev_qlx, dev_qly, dev_qgx, dev_qgy,
                                       dev_phi,
                                       dev_rsd_l, dev_rsd_l,
                                       K0,
                                       nx, ny, Lx, Ly);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at SetIC\n", err);

    Compute_S_GPU();
}


void Problem::Compute_Q_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Q<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg, dev_Sl,
                                           dev_Kx, dev_Ky,
                                           dev_Krlx, dev_Krly, dev_Krgx, dev_Krgy,
                                           dev_qlx, dev_qly, dev_qgx, dev_qgy,
                                           rhol, rhog,
                                           mul, mug,
                                           g,
                                           nx, ny, dx, dy);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Q\n", err);
}

void Problem::Compute_K_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    //kernel_Compute_K<<<dimGrid,dimBlock>>>(dev_Pl, dev_Kx, dev_Ky,
    //                                       K0, gamma, Pt, P0,
    //                                       nx, ny);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at K\n", err);
}

void Problem::Compute_S_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);

    kernel_Compute_S<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg, dev_Sl,
                                           rhol, g,
                                           vg_a, vg_n, vg_m,
                                           nx, ny);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at S\n", err);
}

void Problem::Compute_Kr_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);

    kernel_Compute_Kr<<<dimGrid,dimBlock>>>(dev_Sl,
                                           dev_Krlx, dev_Krly, dev_Krgx, dev_Krgy,
                                           vg_m,
                                           nx, ny);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Kr\n", err);
}

void Problem::Update_P_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_P<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg,
                                          dev_Sl, dev_Sl_old,
                                          dev_qlx, dev_qly, dev_qgx, dev_qgy,
                                          dev_phi, dev_phi_old,
                                          dev_rsd_l, dev_rsd_g,
                                          rhol, rhog,
                                          nx, ny, dx, dy, dt);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at P\n", err);
}

void Problem::Update_P_impl_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_P_impl<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg, dev_Pc, dev_Pl_old, dev_Pg_old,
                                               dev_Sl, dev_Sl_old,
                                               dev_qlx, dev_qly, dev_qgx, dev_qgy,
                                               dev_Krlx, dev_Krly, dev_Krgx, dev_Krgy,
                                               dev_phi, dev_phi_old,
                                               dev_rsd_l, dev_rsd_g,
                                               mul, mug,
                                               rhol, rhog,
                                               vg_a, vg_n, vg_m,
                                               nx, ny, dx, dy, dt);
    cudaError_t err = cudaGetLastError();
    if(err != 0){
        printf("Error %x at P_impl\n", err);
        fflush(stdout);
        exit(0);
    }
}

void Problem::Update_Poro_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Poro<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg,
                                             dev_Pl_old, dev_Pg_old,
                                             dev_Sl, dev_Sl_old,
                                             dev_phi, c_phi, nx, ny);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Poro\n", err);
}

void Problem::Count_Mass_GPU()
{
    //return;
    DAT sum = 0.0;

    cublasHandle_t handle;
    cublasStatus_t stat;
    cublasCreate(&handle);
    stat = cublasDasum(handle, nx*ny, dev_Sl, 1, &sum);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Sum failed\n");
    cublasDestroy(handle);
    DAT mass_new = sum*dx*dy * 0.16 * rhol;

    cublasCreate(&handle);
    stat = cublasDasum(handle, nx*ny, dev_Sl_old, 1, &sum);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Sum failed\n");
    cublasDestroy(handle);
    DAT mass_old = sum*dx*dy * 0.16 * rhol;

    //printf("Liquid mass is %e kg, change is %lf %%\n", mass_new, (mass_new-mass_l)/mass_l*100);
    //DAT dt = 220*60/1e0;

    DAT change       = mass_new - mass_old;
    DAT change_prcnt = (mass_new-mass_old)/mass_old*100;
    DAT change_expct = FLUX*dt*Lx;
    printf("Mass change is %e kg (%2.1lf%%), expected %e kg, diff = %e kg\n", change,
                                                                change_prcnt,
                                                                change_expct,
                                                                change - change_expct);

    flog << "Mass change is " << mass_new-mass_old << " kg (" << (mass_new-mass_old)/mass_old*100 << "%)" << endl;
    mass_l = mass_new;
}

void Problem::H_Substep_GPU()
{
    printf("Flow\n");
    fflush(stdout);
    DAT err_l = 1, err_g = 1, err_l_old, err_g_old, err_l_0 = -1, err_g_0 = -1;
    int nout;
    if(max(nx,ny) < 256)
        nout = 1000;
    else
        nout = 10000;
    for(int nit = 1; nit <= niter; nit++){
        //printf("iter %d\n", nit);
        fflush(stdout);
        //Compute_K_GPU();
        Compute_S_GPU();
        Compute_Kr_GPU();
        Compute_Q_GPU();
        //Update_P_GPU();
        Update_P_impl_GPU();
        if(nit%1000 == 0 || nit == 1 || nit == 2){
            err_l_old = err_l;
            err_g_old = err_g;
            FindMax(dev_rsd_l, &err_l, nx*ny);
            FindMax(dev_rsd_g, &err_g, nx*ny);
            if(nit == 1){
                err_l_0 = err_l;
                err_g_0 = err_g;
            }
            printf("iter %d: r_l = %e, r_g = %e\n", nit, err_l, err_g);
            //printf("iter %d: dr_l = %e, dr_g = %e\n", nit, err_l-err_l_old, err_g-err_g_old);
            fflush(stdout);
            if(err_l<eps_a_h && err_g<eps_a_h){
                printf("Flow converged by abs.crit. in %d it.: r_l = %e, r_g = %e\n", nit, err_l, err_g);
                fflush(stdout);
                flog << "Converged in " << nit << endl;
                break;
            }
            if(err_l < eps_r_h * err_l_0 || err_g < eps_r_h * err_g_0){
                printf("Flow converged by rel.crit. in %d it.: r_l = %e, r_g = %e\n", nit, err_l, err_g);
                fflush(stdout);
                flog << "Converged in " << nit << endl;
                break;
            }
            if(fabs(err_l-err_l_old) < 1e-15 && fabs(err_g-err_g_old) < 1e-15){
                printf("Flow converged by change crit. in %d it.: r_l = %e, r_g = %e\n", nit, err_l, err_g);
                fflush(stdout);
                flog << "Converged in " << nit << endl;
                break;
            }
            string name = respath + "/isol" + to_string(nit) + ".vtk";
            SaveVTK_GPU(name);
            //Count_Mass_GPU();
        }
    }
    //Update_Poro_GPU();
    Count_Mass_GPU();
}

void Problem::SolveOnGPU()
{
    cudaEvent_t tbeg, tend;
    cudaEventCreate(&tbeg);
    cudaEventCreate(&tend);
    cudaEventRecord(tbeg);
    cudaMalloc((void**)&dev_Pl,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pl_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pg,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pg_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pc,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sl,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sl_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_qlx,    sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_qly,    sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_qgx,    sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_qgy,    sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Kx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Ky,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Krlx,   sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Krly,   sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Krgx,   sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Krgy,   sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_phi,    sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_phi_old,sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_rsd_l,  sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_rsd_g,  sizeof(DAT) * nx*ny);
    cudaEventRecord(tbeg);

    printf("Allocated on GPU\n");

    // Still needed for VTK saving
    Pl      = new DAT[nx*ny];
    Pg      = new DAT[nx*ny];
    Sl      = new DAT[nx*ny];
    qlx      = new DAT[(nx+1)*ny];
    qly      = new DAT[nx*(ny+1)];
    Kx      = new DAT[(nx+1)*ny];
    Ky      = new DAT[nx*(ny+1)];
    phi     = new DAT[nx*ny];
    rsd_l   = new DAT[nx*ny];
    rsd_g   = new DAT[nx*ny];

    SetIC_GPU();
    cudaDeviceSynchronize();
    Count_Mass_GPU();

//    for(int i = 0; i < nx+1; i++){
//        for(int j = 0; j < nx; j++){
//            Kx[i+j*(nx+1)] = (1.+10*rand()/RAND_MAX)*K0;
//            if(abs(i-j) < 3)
//                Kx[i+j*(nx+1)] = 100*K0;
//        }
//    }
//    for(int i = 0; i < nx; i++){
//        for(int j = 0; j < nx+1; j++){
//            Ky[i+j*nx] = (1.+10*rand()/RAND_MAX)*K0;
//            if(abs(i-j) < 3)
//                Ky[i+j*nx] = 50*K0;
//        }
//    }
//    cudaMemcpy(dev_Kx, Kx, sizeof(DAT)*(nx+1)*ny, cudaMemcpyHostToDevice);
//    cudaMemcpy(dev_Ky, Ky, sizeof(DAT)*nx*(ny+1), cudaMemcpyHostToDevice);

//    for(int i = 0; i < nx; i++){
//        for(int j = 0; j < nx; j++){
//            phi[i+j*nx] = (0.1 + (double) rand() / (RAND_MAX))*0.16;
//        }
//    }
//    cudaMemcpy(dev_phi, phi, sizeof(DAT)*nx*ny, cudaMemcpyHostToDevice);

    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME STEP %d, T = %lf s =======\n", it, it*dt);
        flog << "\n\n =======  TIME STEP " << it <<", T = " << it*dt << " s =======\n";
        if(do_mech)
;//            M_Substep_GPU();
        cudaMemcpy(dev_Pl_old,  dev_Pl,  sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Pg_old,  dev_Pg,  sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Sl_old,  dev_Sl,  sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_phi_old, dev_phi, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        H_Substep_GPU();
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
        //SaveDAT_GPU(it);
    }

    cudaFree(dev_Pl);
    cudaFree(dev_Pl_old);
    cudaFree(dev_Pg);
    cudaFree(dev_Pg_old);
    cudaFree(dev_Pc);
    cudaFree(dev_Sl);
    cudaFree(dev_Sl_old);
    cudaFree(dev_qlx);
    cudaFree(dev_qly);
    cudaFree(dev_qgx);
    cudaFree(dev_qgy);
    cudaFree(dev_Kx);
    cudaFree(dev_Ky);
    cudaFree(dev_Krlx);
    cudaFree(dev_Krly);
    cudaFree(dev_Krgx);
    cudaFree(dev_Krgy);
    cudaFree(dev_phi);
    cudaFree(dev_phi_old);
    cudaFree(dev_rsd_l);
    cudaFree(dev_rsd_g);

    cudaEventRecord(tend);
    cudaEventSynchronize(tend);

    float comptime = 0.0;
    cudaEventElapsedTime(&comptime, tbeg, tend);
    printf("\nComputation time = %f s\n", comptime/1e3);

    delete [] Pl;
    delete [] Pg;
    delete [] Sl;
    delete [] qlx;
    delete [] qly;
    delete [] Kx;
    delete [] Ky;
    delete [] phi;
    delete [] rsd_l;
    delete [] rsd_g;
}


__global__ void kernel_SetIC(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Sl,
                             DAT *Kx, DAT *Ky,
                             DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                             DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                             DAT *phi,
                             DAT *rsd_l, DAT *rsd_g,
                             const DAT K0,
                             const int nx, const int ny,
                             const DAT Lx, const DAT Ly
                             )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    const DAT dx = Lx/nx, dy = Ly/ny;

    DAT x = (i+0.5)*dx, y = (j+0.5)*dy;
    // Cell variables
    if(i >= 0 && i < nx && j >= 0 && j < ny){
        if(sqrt((Lx/2.0-x)*(Lx/2.0-x) + (Ly/2.0-y)*(Ly/2.0-y)) < 0.001)
            Pl[i+j*nx] = 11e6;//8.1e6;//11e6;
        else
            Pl[i+j*nx] = 8e6;

        //Pl[i+j*nx] = 8e6;
        Pg[i+j*nx] = 10e6;
        Pc[i+j*nx] = Pg[i+j*nx] - Pl[i+j*nx];
        phi[i+j*nx] = 0;//.16; shit
        rsd_l[i+j*nx] = 0.0;
        rsd_g[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >= 0 && i <= nx && j >= 0 && j < ny){
        int ind = i+j*(nx+1);
        qlx[ind]  = 0.0;
        qgx[ind]  = 0.0;
        Kx[ind]   = K0;
        Krlx[ind] = 1.0;
        Krgx[ind] = 1.0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >= 0 && i < nx && j >= 0 && j <= ny){
        int ind = i+j*nx;
        qly[ind]  = 0.0;
        qgy[ind]  = 0.0;
        Ky[ind]   = K0;
        Krly[ind] = 1.0;
        Krgy[ind] = 1.0;
    }
}

__global__ void kernel_Compute_S(DAT *Pl, DAT *Pg, DAT *Sl,
                                 const DAT rhol, const DAT g,
                                 const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                 const int nx, const int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        // Pl = Pg - Pc
        // Pc = Pg - Pl

        const DAT p = 1.0, M = 1.0, Pe = 10e6;

        DAT Pc = Pg[i+j*nx] - Pl[i+j*nx];

        if(Pc <= 0.0)
        //if(Pc >= Pe)
            Sl[i+j*nx] = 1.0;
        else{
            Sl[i+j*nx] = pow(1.0 + pow(vg_a/rhol/g*Pc, vg_n), -vg_m); // S = (1 + P^n)^(-m)
            //
            //Sl[i+j*nx] = 1. / (1. + pow(Pc+Pe, 1./p));
        }
    }
}

__global__ void kernel_Compute_Kr(DAT *Sl,
                                  DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                  const DAT vg_m,
                                  const int nx, const int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        // Simple central approximation
        DAT S = 0.5*(Sl[i+j*nx]+Sl[i-1+j*nx]);
        //DAT S = max(Sl[i+j*nx], Sl[i-1+j*nx]);

        Krlx[i+j*(nx+1)] = pow(S,2.0);
        Krgx[i+j*(nx+1)] = pow(1.-S,2.0);

        //Krlx[i+j*(nx+1)] = sqrt(S) * pow(1.-pow(1.-(pow(S,1./vg_m)), vg_m), 2.);
        //Krgx[i+j*(nx+1)] = 1.3978-3.7694*S+12.709*S*S-20.642*S*S*S+10.309*S*S*S*S;
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        // Simple central approximation
        DAT S = 0.5*(Sl[i+j*nx]+Sl[i+(j-1)*nx]);
        //DAT S = max(Sl[i+j*nx], Sl[i+(j-1)*nx]);

        Krly[i+j*nx] = pow(S,2.0);
        Krgy[i+j*nx] = pow(1.-S,2.0);

        //Krly[i+j*nx] = sqrt(S) * pow(1.-pow(1.-(pow(S,1./vg_m)), vg_m), 2.);
        //if(S < 0.99)
        //    Krly[i+j*nx] = 1e-8;
        //Krgy[i+j*nx] = 1.3978-3.7694*S+12.709*S*S-20.642*S*S*S+10.309*S*S*S*S;
    }
}


__global__ void kernel_Compute_Q(DAT *Pl, DAT *Pg,
                                 DAT *Sl,
                                 DAT *Kx, DAT *Ky,
                                 DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                 DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                 const DAT rhol, const DAT rhog,
                                 const DAT mul, const DAT mug,
                                 const DAT g,
                                 const int nx, const int ny,
                                 const DAT dx, const DAT dy)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal fluxes
        int ind = i+j*(nx+1);
        qlx[ind] = -rhol*Kx[ind]/mul*Krlx[ind]*((Pl[i+j*nx]-Pl[i-1+j*nx])/dx);
        qgx[ind] = -rhog*Kx[ind]/mug*Krgx[ind]*((Pg[i+j*nx]-Pg[i-1+j*nx])/dx);
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal fluxes
        int ind = i+j*nx;
        qly[ind] = -rhol*Ky[ind]/mul*Krly[ind]*((Pl[i+j*nx]-Pl[i+(j-1)*nx])/dy + 0*rhol*g);
        qgy[ind] = -rhog*Ky[ind]/mug*Krgy[ind]*((Pg[i+j*nx]-Pg[i+(j-1)*nx])/dy + 0*rhog*g);
    }

    // Inflow BC at the lower side
    if(i >= 0 && i <= nx-1 && j == 0){
        DAT x = (i+0.5)*dx;
        //if(x > 0.012*0.25 && x < 0.012*0.75)
        qly[i+j*nx] = FLUX;
        //qly[i+j*nx] = -rhol*1/mul*Krly[i+j*nx]*((Pl[i+j*nx]-0)/dy + 0*rhol*g);
    }


    if(j >= 0 && j <= ny-1 && i == 0){
        //qlx[i+j*(nx+1)] = -rhol*1e0/mul*((Pl[i+j*nx]-1)/dx);
    }


    if(i >= 0 && i <= nx-1 && j == ny){
        //qly[i+j*nx] = -rhol*1e-18/mul*1.0*((0e6-Pl[i+(j-1)*nx])/dy + 0*rhol*g);
        //qgy[i+j*nx] = -rhol*1e-18/mug*1.0*((8e6-Pg[i+(j-1)*nx])/dy + 0*rhog*g);
        if(i == 0)
            ;//printf("flux = %e, P = %e, dP = %e\n", qly[i+j*nx],
             //                                    Pl[i+(j-1)*nx],
             //                                    Ky[i+j*nx]*(10e16-Pl[i+(j-1)*nx])/dy);
    }
}

__global__ void kernel_Compute_K(DAT *Pl,
                                 DAT *Kx, DAT *Ky,
                                 const DAT K0, const DAT gamma,
                                 const DAT Pt, const DAT P0,
                                 const int nx, const int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        // Upwind approach
        DAT Pupw = max(Pl[i+j*nx], Pl[i+1+j*nx]);
        //DAT Pupw = 0.5*(Pf[i+j*nx] + Pf[i+1+j*nx]);
        Kx[i+j*(nx+1)] = K0 * exp(-gamma*(Pt-Pupw-P0));
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        // Upwind
        DAT Pupw = max(Pl[i+j*nx], Pl[i+(j+1)*nx]);
        //DAT Pupw = 0.5*(Pl[i+j*nx] + Pl[i+(j+1)*nx]);
        Ky[i+j*nx] = K0 * exp(-gamma*(Pt-Pupw-P0));
    }
}

__global__ void kernel_Update_P(DAT *Pl, DAT *Pg,
                                DAT *Sl, DAT *Sl_old,
                                DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                DAT *phi, DAT *phi_old,
                                DAT *rsd_l, DAT *rsd_g,
                                const DAT rhol, const DAT rhog,
                                const int nx, const int ny,
                                const DAT dx, const DAT dy, const DAT dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        rsd_l[ind] = rhol * (phi[ind]*Sl[ind] - phi_old[ind]*Sl_old[ind])/dt
                 + (qlx[i+1+j*(nx+1)] - qlx[i+j*(nx+1)])/dx
                 + (qly[i+(j+1)*nx]   - qly[i+j*nx])/dy;

        rsd_g[ind] = rhog * (phi[ind]*Sl[ind] - phi_old[ind]*Sl_old[ind])/dt
                 + (qgx[i+1+j*(nx+1)] - qgx[i+j*(nx+1)])/dx
                 + (qgy[i+(j+1)*nx]   - qgy[i+j*nx])/dy;

        DAT D    = max(rhol,rhog)*1e-18/1e-3;//min(mul,mug);

        DAT dt_h = min(dx*dx,dy*dy) / D / 4.1;//1e3;
           dt_h *= 1e-3;

        Pl[ind]  -= rsd_l[ind] * dt_h;
        Pg[ind]  -= rsd_g[ind] * dt_h;

        rsd_l[ind] = fabs(rsd_l[ind]);
        rsd_g[ind] = fabs(rsd_g[ind]);
    }
}

__global__ void kernel_Update_P_impl(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Pl_old, DAT *Pg_old,
                                     DAT *Sl, DAT *Sl_old,
                                     DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                     DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                     DAT *phi, DAT *phi_old,
                                     DAT *rsd_l, DAT *rsd_g,
                                     const DAT mul, const DAT mug,
                                     const DAT rhol, const DAT rhog,
                                     const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                     const int nx, const int ny,
                                     const DAT dx, const DAT dy, const DAT dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        DAT PHI = 0.16;

        // Compute residual
        DAT div_ql = (qlx[i+1+j*(nx+1)] - qlx[i+j*(nx+1)])/dx
                   + (qly[i+(j+1)*nx]   - qly[i+j*nx])/dy;
        DAT div_qg = (qgx[i+1+j*(nx+1)] - qgx[i+j*(nx+1)])/dx
                   + (qgy[i+(j+1)*nx]   - qgy[i+j*nx])/dy;

        DAT aPc    = vg_a/rhol/9.81*(Pg[ind] - Pl[ind]);
        DAT dSldPc;
        if(aPc < 0.0){
            dSldPc = 0.0;
            //return; ///// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        }
        else{
            dSldPc = -vg_m*vg_n * pow(aPc,vg_n-1.) * pow(1. + pow(aPc,vg_n), -vg_m-1.); // < 0
            //dSldPc = -dSldPc;
        }



        rsd_l[ind] =  PHI*rhol * (Sl[ind] - Sl_old[ind])/dt + div_ql;
        rsd_g[ind] = -PHI*rhog * (Sl[ind] - Sl_old[ind])/dt + div_qg;

        DAT Krl = max(max(Krlx[i+j*(nx+1)], Krlx[i+j*(nx+1)]), max(Krly[i+j*nx], Krly[i+(j+1)*nx]));
        DAT Krg = max(max(Krgx[i+j*(nx+1)], Krgx[i+j*(nx+1)]), max(Krgy[i+j*nx], Krgy[i+(j+1)*nx]));

        // Determine pseudo-transient step
        DAT D;    // "Diffusion coefficient" for the cell
        DAT dtau; // Pseudo-transient step
        if(fabs(dSldPc) > 1e-10 && false){
            D    = 700*1e-18/min(mul,mug) / PHI/rhol;// / fabs(dSldPc);// * max(rhol,rhog) * max(Krl,Krg);
            dtau = 1./(4.1*D/min(dx*dx,dy*dy) + 1./dt);
            //dtau = min(dx*dx,dy*dy) / D / 4.1;
            //dtau *= 100;
        }
        else{
            D    = max(rhol,rhog)*1e-18/min(mul,mug) / PHI/rhol;// * max(Krl,Krg);
            dtau = min(dx*dx,dy*dy) / D / 4.1;
            dtau *= 1e-6;
            //dtau *= fabs(dSldPc) * 1e-1;
        }
        if(dtau < 1e-15)
            return;

//        DAT Pc_old = Pg_old[ind] - Pl_old[ind];
        DAT cc = 1.0, cg = 1.0, cl = 1.0;

//        DAT dS = (Sl[ind] - Sl_old[ind])/dt;
//        // a*Pl = b + c*Pg
//        // d*Pg = e + f*Pl
//        DAT PAR = -0*(1.-1./nx);
//        DAT a  = -dSldPc/dt + cl/dtau;
//        DAT b  = Pl[ind]*a - dS - div_ql/PHI/rhol + PAR*phi[ind] + dSldPc*Pg[ind]/dt;
//        DAT c  = -dSldPc/dt;
//        DAT d  = -dSldPc/dt + cg/dtau;
//        DAT e  = Pg[ind]*d + dS - div_qg/PHI/rhog + PAR*Pc[ind] + dSldPc*Pl[ind]/dt;
//        DAT f  = c;

//        Pl[ind] = (b*d + c*e)/(a*d - c*f);
//        Pg[ind] = (a*e + b*f)/(a*d - c*f);


//        phi[ind]   = (Sl[ind] - Sl_old[ind])/dt;//rsd_l[ind];
//        Pc[ind]    = rsd_g[ind];

        DAT cC = 1.0, Slp = Sl[ind];
        //Sl[ind] = 0.22;

        //Sl[ind] -= rsd_l[ind]*dtau;

//        if(Sl[ind] > 1.0)
//            Sl[ind] = 1.0;
//        if(Sl[ind] < 1e-8)
//            Sl[ind] = 1e-8;

        for(; dtau > 1e-18; dtau *= 0.1){
            Sl[ind] = cC*Sl[ind]/dtau + Sl_old[ind]/dt - div_ql/PHI/rhol;
            Sl[ind] = Sl[ind]/(1./dt + cC*1./dtau);
            if(Sl[ind] > 1.0 || Sl[ind] < 0.0 || isinf(Sl[ind]) || isnan(Sl[ind])){
                //printf("Bad Sl = %e at cell (%d %d), Slp = %e, Sl_old = %e, divq = %e, dtau = %e\n", Sl[ind], i, j, Slp, Sl_old[ind], div_ql/PHI/rhol, dtau);
                Sl[ind] = Slp;
            }
            else
                break;

        }
        phi[ind] = dtau;

        Pg[ind] = Pg[ind] + dtau * ((Sl[ind] - Sl_old[ind])/dt - div_qg/PHI/rhog);

        DAT Pcc = rhol*9.81/vg_a * pow(pow(Sl[ind],-1./vg_m) - 1., 1./vg_n);
        if(isnan(Pcc) || isinf(Pcc)){
            printf("Bad Pcc at cell (%d %d), Sl = %e\n", i, j, Sl[ind]);
            //exit(0);
            __trap();
        }
        Pl[ind] = Pg[ind] - Pcc;



        // END
        rsd_l[ind] = fabs(rsd_l[ind]);
        rsd_g[ind] = fabs(rsd_g[ind]);
    }
}

__global__ void kernel_Update_P_diff(DAT *Pl, DAT *Pg, DAT *Pc, DAT *Pl_old, DAT *Pg_old,
                                     DAT *Sl, DAT *Sl_old,
                                     DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                                     DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                                     DAT *phi, DAT *phi_old,
                                     DAT *rsd_l, DAT *rsd_g,
                                     const DAT mul, const DAT mug,
                                     const DAT rhol, const DAT rhog,
                                     const DAT vg_a, const DAT vg_n, const DAT vg_m,
                                     const int nx, const int ny,
                                     const DAT dx, const DAT dy, const DAT dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        // Compute residual
        DAT div_ql = (qlx[i+1+j*(nx+1)] - qlx[i+j*(nx+1)])/dx
                   + (qly[i+(j+1)*nx]   - qly[i+j*nx])/dy;




        rsd_l[ind] = 0*(Pl[ind] - Pl_old[ind])/dt + div_ql;

        //rsd_l[ind] = fabs(rsd_l[ind]);
        rsd_g[ind] = 0;


        // Determine pseudo-transient step
        DAT D;    // "Diffusion coefficient" for the cell
        DAT dtau; // Pseudo-transient step

        D    = 1;//700*1e-18/min(mul,mug);


        //dtau = 1./(4.1*D/min(dx*dx,dy*dy) + 1./dt) * 1e0;
        //dtau = 1./(4.1*D/min(dx*dx,dy*dy)) * 1e-2;
        dtau = 1./(8.1*D/min(dx,dy));

        // 'Implicit' update
        //Pl[ind] = Pl[ind]/dtau - div_ql - 0*(1.-5./nx)*phi[ind] + 0*Pl_old[ind]/dt;
        //Pl[ind] = Pl[ind]/(0*1./dt + 1./dtau);

        //Pl[ind]    = Pl[ind] - dtau * (rsd_l[ind] + (1.-1./nx)*phi[ind]);

//        DAT qxl = 0.0, qxr = 0.0, qyl = 0.0, qyu = 0.0;

//        if(i > 0)
//            qxl = -D*(Pl[i+j*nx]-Pl[i-1+j*nx])/dx;
//        else
//            qxl = -D*(Pl[i+1+j*nx]-9e6)/dx;

//        if(i < nx-1)
//            qxr = -D*(Pl[i+1+j*nx]-Pl[i+j*nx])/dx;
//        else
//            qxr = -D*(7e6-Pl[i+j*nx])/dx;

//        if(j > 0)
//            qyl = -D*(Pl[i+j*nx]-Pl[i+(j-1)*nx])/dx;
//        else
//            qyl = -D*(Pl[i+j*nx]-7e6)/dx;

//        if(j < ny-1)
//            qyu = -D*(Pl[i+(j+1)*nx]-Pl[i+j*nx])/dx;
//        else
//            qyu = -D*(9e6-Pl[i+j*nx])/dx;
//        DAT div_q = (qxr-qxl)/dx + (qyu-qyl)/dy;




        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //! Pc contains rsd_old
        //! phi contains rsd_old2
        //! Pl_old contains P_old
        //! Pg_old contains P_old2
        Pg_old[ind]= Pl_old[ind];
        Pl_old[ind]= Pl[ind];
        phi[ind]   = Pc[ind];
        Pc[ind]    = rsd_l[ind];
        rsd_l[ind] = div_ql;

        DAT A = 1.-2./nx;

        //Pl[ind] = Pl[ind] - dtau * (div_ql + 0*(1.-0./nx)*phi[ind]);
        //Pl[ind] = 2*Pl[ind] - phi[ind] - dtau*dtau * div_ql;

        DAT P_old2 = Pg_old[ind];
        Pl[ind] = (-P_old2 + Pl[ind]*(2.+A*dtau) - dtau*dtau * div_ql)/(1.+A*dtau);
        rsd_l[ind] = fabs(rsd_l[ind]);
    }
}

__global__ void kernel_Update_Poro(DAT *Pl, DAT *Pg,
                                   DAT *Pl_old, DAT *Pg_old,
                                   DAT *Sl, DAT *Sl_old,
                                   DAT *phi,
                                   const DAT c_phi,
                                   const int nx, const int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        DAT Pf     = Sl[ind]    *Pl[ind]     + (1.0-Sl[ind])    *Pg[ind];
        DAT Pf_old = Sl_old[ind]*Pl_old[ind] + (1.0-Sl_old[ind])*Pg_old[ind];

        // Explicit update
        //phi[ind] += c_phi*phi[ind]*(Pf[ind] - Pf_old[ind]);

        // Implicit update
        phi[ind] /= (1.0 - c_phi*(Pf-Pf_old));
    }
}

void Problem::SaveVTK_GPU(std::string path)
{
    //return;
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pl,  dev_Pl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Pg,  dev_Pg, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sl,  dev_Sl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qlx, dev_qlx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qly, dev_qly, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kx,  dev_Kx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ky,  dev_Ky, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(phi, dev_phi, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(rsd_l,  dev_rsd_l, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(rsd_g,  dev_rsd_g, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);

    SaveVTK(path);
}

void Problem::SaveDAT_GPU(int stepnum)
{
    cudaMemcpy(Pl, dev_Pl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Kx, dev_Kx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ky, dev_Ky, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    std::string path = "C:\\Users\\Denis\\Documents\\msu_thmc\\MATLAB\\res_gpu\\shale_1phase\\";
    FILE *f;
    std::string fname;

    fname = path + "Pl" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Pl, sizeof(DAT), ny*nx, f);
    fclose(f);

    fname = path + "Ky" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Ky, sizeof(DAT), (ny+1)*nx, f);
    fclose(f);

    fname = path + "Kx" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Kx, sizeof(DAT), (nx+1)*ny, f);
    fclose(f);
}
