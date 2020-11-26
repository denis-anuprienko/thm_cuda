#include "header.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define BLOCK_DIM 16
#define dt_h      1e0
//#define dt_m      1e-6
//__device__ DAT dt_m;

//__global__ void set_dtm(DAT val)
//{
//    dt_m = val;
//}

DAT DTAU_; // Global PT step for coupled problem
DAT DTAU_M;

#define FLUX 1e-2

void FindMax(DAT *dev_arr, DAT *max, int size);

__global__ void kernel_SetIC(DAT *Txx, DAT *Tyy, DAT *Txy, DAT *Vx, DAT *Vy, DAT *Ux, DAT *Uy,
                             DAT *rsd_m_x, DAT *rsd_m_y,
                             DAT *Pw, DAT *Sw, DAT *qx, DAT *qy, DAT *Krx, DAT *Kry,
                             DAT *rsd_h, const DAT rhow, const DAT vg_a, const DAT vg_m, const DAT vg_n,
                             const int nx, const int ny, const DAT Lx, const DAT Ly);
__global__ void kernel_Compute_Sw(DAT *Pw, DAT *Sw, const int nx, const int ny,
                             const DAT rhow, const DAT g,
                             const DAT vg_a, const DAT vg_m, const DAT vg_n);
__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pw, DAT *Krx, DAT *Kry,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT K,  const DAT rhow, const DAT muw, const DAT g);
__global__ void kernel_Compute_Kr(DAT *qx, DAT *qy, DAT *Pw, DAT *Sw, DAT *Krx, DAT *Kry,
                                 const int nx, const int ny, const DAT vg_m);

__global__ void kernel_Update_Pw(DAT *rsd, DAT *Pw, DAT *Sw, DAT *Pw_old, DAT *Sw_old,
                                 DAT *qx, DAT *qy, DAT *Ux, DAT *Uy, DAT *Ux_old, DAT *Uy_old, const int nx, const int ny,
                                 const DAT dx, const DAT dy, const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor);

__global__ void kernel_Update_Sw(DAT *rsd, DAT *Pw, DAT *Sw, DAT *Pw_old, DAT *Sw_old,
                                 DAT *qx, DAT *qy, DAT *Ux, DAT *Uy, DAT *Ux_old, DAT *Uy_old, const int nx, const int ny,
                                 const DAT dx, const DAT dy, const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor, const DAT muw, const DAT K, const DAT vg_a, const DAT vg_m, const DAT vg_n, const DAT DTAU);

__global__ void kernel_Update_V(DAT *Vx, DAT *Vy, DAT *Txx, DAT *Tyy, DAT *Txy, DAT *Pw, DAT *Sw,
                                const int nx, const int ny, const DAT dx, const DAT dy,
                                const DAT rho_s, const DAT g, const DAT dt_m);

__global__ void kernel_Update_U(DAT *Ux, DAT *Uy, DAT *Vx, DAT *Vy, const int nx, const int ny, const DAT dt_m);
__global__ void kernel_Update_Stress(DAT *Txx, DAT *Tyy, DAT *Txy, DAT *Vx, DAT *Vy, DAT *Pw, DAT *Sw, DAT *Sw_old, DAT *rsd_m_x, DAT *rsd_m_y, const int nx, const int ny,
                                 const DAT dx,  const DAT dy, const DAT dt,
                                 const DAT rho_s, const DAT g, const DAT mu, const DAT lam, const DAT dt_m);

void FindMax(DAT *dev_arr, DAT *max, int size)
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    cublasCreate(&handle);

    int maxind = 0;
    stat = cublasIdamax(handle, size, dev_arr, 1, &maxind);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Max failed\n");


    cudaMemcpy(max, dev_arr+maxind-1, sizeof(DAT), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
}

void Problem::Count_Mass_GPU()
{
    DAT mass_new = 0.0, mass_old = 0.0; // total water mass

    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Mul\n", err);
    cublasHandle_t handle;
    cublasStatus_t stat;
    cublasCreate(&handle);

    stat = cublasDasum(handle, nx*ny, dev_Sw, 1, &mass_new);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Sum failed\n");
    mass_new *= dx*dy * rhow;

    stat = cublasDasum(handle, nx*ny, dev_Sw_old, 1, &mass_old);
    if (stat != CUBLAS_STATUS_SUCCESS)
        printf("Sum failed\n");
    mass_old *= dx*dy * rhow;

    cublasDestroy(handle);

    DAT change       = mass_new - mass_old;
    DAT change_prcnt = (mass_new-mass_old)/mass_old*100;
    DAT change_expct = FLUX*dt*Lx;
    printf("Mass change is %e kg (%2.2lf%%), expected %e kg, diff = %e kg\n", change,
                                                                change_prcnt,
                                                                change_expct,
                                                                change - change_expct);
    printf("Ratios: %e %e dx = %e dy = %e\n", change/change_expct, change_expct/change, dx, dy);
}

void Problem::SetIC_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    printf("Launching %dx%d blocks of %dx%d threads\n", (nx+1+dimBlock.x-1)/dimBlock.x,
           (ny+1+dimBlock.y-1)/dimBlock.y, BLOCK_DIM, BLOCK_DIM);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Txx, dev_Tyy, dev_Txy,
                                       dev_Vx, dev_Vy, dev_Ux, dev_Uy,
                                       dev_rsd_m_x, dev_rsd_m_y,
                                       dev_Pw, dev_Sw, dev_qx, dev_qy, dev_Krx, dev_Kry, dev_rsd_h,
                                       rhow, vg_a, vg_m, vg_n,
                                       nx, ny, Lx, Ly);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at SetIC\n", err);
}

void Problem::Update_V_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Update_V<<<dimGrid,dimBlock>>>(dev_Vx, dev_Vy, dev_Txx, dev_Tyy, dev_Txy,
                                          dev_Pw, dev_Sw,
                                          nx, ny, dx, dy, rho_s, g, DTAU_);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at V\n", err);
}

void Problem::Update_U_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Update_U<<<dimGrid,dimBlock>>>(dev_Ux, dev_Uy, dev_Vx, dev_Vy, nx, ny, DTAU_M);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at U\n", err);
}

void Problem::Update_Stress_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Stress<<<dimGrid,dimBlock>>>(dev_Txx, dev_Tyy, dev_Txy,
                                               dev_Vx, dev_Vy, dev_Pw, dev_Sw, dev_Sw_old,
                                               dev_rsd_m_x, dev_rsd_m_y,
                                               nx, ny, dx, dy, dt, rho_s, g, mu, lam, DTAU_M);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Stress\n", err);
}

void Problem::Compute_Sw_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Sw<<<dimGrid,dimBlock>>>(dev_Pw, dev_Sw, nx, ny, rhow, g, vg_a, vg_m, vg_n);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x\n", err);
}

void Problem::Compute_Q_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Q<<<dimGrid,dimBlock>>>(dev_qx, dev_qy, dev_Pw, dev_Krx, dev_Kry,
                                           nx, ny, dx, dy, K, rhow, muw, g);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Q\n", err);
}

void Problem::Compute_Kr_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Kr<<<dimGrid,dimBlock>>>(dev_qx, dev_qy, dev_Pw, dev_Sw, dev_Krx, dev_Kry,
                                            nx, ny, vg_m);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Kr\n", err);
}

void Problem::Update_Pw_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Pw<<<dimGrid,dimBlock>>>(dev_rsd_h, dev_Pw, dev_Sw, dev_Pw_old, dev_Sw_old,
                                           dev_qx, dev_qy,
                                           dev_Ux, dev_Uy, dev_Ux_old, dev_Uy_old,
                                           nx, ny, dx, dy, dt,
                                           phi, rhow, sstor);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Pw\n", err);
}

void Problem::Update_Sw_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Sw<<<dimGrid,dimBlock>>>(dev_rsd_h, dev_Pw, dev_Sw, dev_Pw_old, dev_Sw_old,
                                           dev_qx, dev_qy,
                                           dev_Ux, dev_Uy, dev_Ux_old, dev_Uy_old,
                                           nx, ny, dx, dy, dt,
                                           phi, rhow, sstor, muw, K,
                                           vg_a, vg_m, vg_n, DTAU_);
    cudaError_t err = cudaGetLastError();
    if(err != 0){
        printf("Error %x at Sw\n", err);
        exit(0);
    }
}

void Problem::M_Substep_GPU()
{
    printf("Mechanics\n");
    fflush(stdout);
    for(int nit = 1; nit <= 100000; nit++){
        Update_V_GPU();
        Update_U_GPU();
        Update_Stress_GPU();
        if(nit%10000 == 0 || nit == 1 || nit == 1000){
            DAT err_m_x = 0.;
            DAT err_m_y = 0.;
//            cudaMemcpy(rsd_m_x, dev_rsd_m_x, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
//            cudaMemcpy(rsd_m_y, dev_rsd_m_y, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
//            for(int i = 0; i < nx*ny; i++){
//                if(fabs(rsd_m_x[i]) > err_m_x)
//                    err_m_x = fabs(rsd_m_x[i]);
//                if(fabs(rsd_m_y[i]) > err_m_y)
//                    err_m_y = fabs(rsd_m_y[i]);

//                if(isinf(rsd_m_x[i]) || isnan(rsd_m_x[i]) || isinf(rsd_m_y[i]) || isnan(rsd_m_y[i])){
//                    printf("Bad value, iter %d", nit);
//                    exit(0);
//                }
//            }
            FindMax(dev_rsd_m_x, &err_m_x, nx*ny);
            FindMax(dev_rsd_m_y, &err_m_y, nx*ny);
            printf("iter %d: r_m_x = %e, r_m_y = %e\n", nit, err_m_x, err_m_y);
            fflush(stdout);
            if(err_m_x < eps_a_m && err_m_y < eps_a_m){
                printf("Mechanics converged in %d it.: r_m_x = %e, r_m_y = %e\n", nit, err_m_x, err_m_y);
                break;
            }
        }
    }

}

void Problem::H_Substep_GPU()
{
    printf("Flow\n");
    fflush(stdout);
    DAT flag, *dev_flag;
    DAT err0;
    cudaMalloc((void**)&dev_flag, sizeof(DAT));
    cudaMemset(dev_flag,0,sizeof(DAT));
    for(int nit = 1; nit <= niter; nit++){
        //Compute_Sw_GPU();
        Compute_Kr_GPU();
        Compute_Q_GPU();
        //Update_Pw_GPU();
        Update_Sw_GPU();
        if(nit%10000 == 0 || nit == 1){
            DAT err = 13;
//            cudaMemcpy(rsd_h, dev_rsd_h, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
//            for(int i = 0; i < nx*ny; i++){
//                if(fabs(rsd_h[i]) > err)
//                    err = fabs(rsd_h[i]);
//                if(isinf(rsd_h[i]) || isnan(rsd_h[i])){
//                    printf("Bad value, iter %d", nit);
//                    exit(0);
//                }
//            }
            FindMax(dev_rsd_h, &err, nx*ny);
            if(nit == 1)
                err0 = err;
            printf("iter %d: r_w = %e\n", nit, err);
            fflush(stdout);
            if(err < eps_a_h){
                printf("Flow converged by abs. crit. in %d it.: r_w = %e\n", nit, err);
                break;
            }
            if(err < eps_r_h*err0){
                printf("Flow converged by rel. crit. in %d it.: r_w = %e\n", nit, err);
                break;
            }
        }
    }
    Count_Mass_GPU();
}

void Problem::HM_Substep_GPU()
{
    printf("Fully coupled poromechanics\n");
    fflush(stdout);

    DAT err_h, err_h0, err_m_x, err_m_y, err_m_x0, err_m_y0;
    bool converged_h = false, converged_m = false;

    DAT D = rhow * K / muw / phi;
    DTAU_  = min(dx*dx,dy*dy) / D / 4.1; // Greater D - smaller  dtau
    DTAU_ *= 1e0;
    DTAU_M = 0.1*sqrt(min(dx*dx,dy*dy) / E / rho_s / 4.2);
    //set_dtm<<<1,1>>>(DTAU_M);

    printf("DTAU_M = %e, E = %e, nu = %e, lam = %e, mu = %e\n", DTAU_M, E, nu, lam, mu);

    for(int nit = 1; nit <= niter; nit++){
        // Mechanics first
        Update_V_GPU();
        Update_U_GPU();
        Update_Stress_GPU();

        // Now flow
        //Compute_Sw_GPU();
//        Compute_Kr_GPU();
//        Compute_Q_GPU();
//        Update_Sw_GPU();
        if(nit%10000 == 0 || nit == 1){
            FindMax(dev_rsd_h, &err_h, nx*ny);
            FindMax(dev_rsd_m_x, &err_m_x, nx*ny);
            FindMax(dev_rsd_m_y, &err_m_y, nx*ny);
            if(nit == 1){
                err_m_x0 = err_m_x;
                err_m_y0 = err_m_y;
                err_h0 = err_h;
            }


            printf("iter %d: r_h = %e, r_m_x = %e, r_m_y = %e\n", nit, err_h, err_m_x, err_m_y);
            fflush(stdout);
            if(err_m_x < eps_a_m && err_m_y < eps_a_m){
                //printf("Mechanics converged by abs.crit. in %d it.: r_m_x = %e, r_m_y = %e\n", nit, err_m_x, err_m_y);
                converged_m = true;
            }
            if(err_m_x < eps_r_m * err_m_x0 && err_m_y < eps_r_m * err_m_y0){
                //printf("Mechanics converged by rel.crit. in %d it.: r_m_x = %e, r_m_y = %e\n", nit, err_m_x, err_m_y);
                converged_m = true;
            }

            if(err_h < eps_a_h){
                //printf("Flow converged by abs. crit. in %d it.: r_w = %e\n", nit, err);
                converged_h = true;
            }
            if(err_h < eps_r_h*err_h0){
                //printf("Flow converged by rel. crit. in %d it.: r_w = %e\n", nit, err);
                converged_h = true;
            }

            if(converged_h && converged_m){
                printf("Coupled problem converged in %d iterations\n", nit);
                break;
            }
        }
    }
    Count_Mass_GPU();
}

void Problem::SolveOnGPU()
{
    cudaEvent_t tbeg, tend;
    cudaEventCreate(&tbeg);
    cudaEventCreate(&tend);
    cudaEventRecord(tbeg);
    cudaMalloc((void**)&dev_Pw,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sw,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pw_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sw_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_qx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_qy,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Krx,    sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Kry,    sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_rsd_h,  sizeof(DAT) * nx*ny);

    cudaMalloc((void**)&dev_Txx,    sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Tyy,    sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Txy,    sizeof(DAT) * (nx+1)*(ny+1));
    cudaMalloc((void**)&dev_Vx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Vy,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Ux,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Uy,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Ux_old, sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Uy_old, sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_rsd_m_x, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_rsd_m_y, sizeof(DAT) * nx*ny);
    cudaEventRecord(tbeg);

    printf("Allocated on GPU\n");

    // Still needed for VTK saving
    Pw      = new DAT[nx*ny];
    Sw      = new DAT[nx*ny];
    qx      = new DAT[(nx+1)*ny];
    qy      = new DAT[nx*(ny+1)];
    rsd_h   = new DAT[nx*ny];
    rsd_m_x = new DAT[nx*ny];
    rsd_m_y = new DAT[nx*ny];

    Tyy     = new DAT[nx*ny];
    Txx     = new DAT[nx*ny];
    Txy     = new DAT[(nx+1)*(ny+1)];
    Ux      = new DAT[(nx+1)*ny];
    Uy      = new DAT[nx*(ny+1)];
    Vx      = new DAT[(nx+1)*ny];
    Vy      = new DAT[nx*(ny+1)];

    std::fill_n(Ux, (nx+1)*ny, 0.0);
    std::fill_n(Uy, nx*(ny+1), 0.0);

    SetIC_GPU();
    cudaDeviceSynchronize();

    // Initial mechanical state
//    if(do_mech)
//        M_Substep_GPU();

    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME STEP %d, TIME = %lf s =======\n", it, it*dt);
        cudaMemcpy(dev_Ux_old, dev_Ux, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Uy_old, dev_Uy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Pw_old, dev_Pw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Sw_old, dev_Sw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
//        if(do_flow)
//            H_Substep_GPU();
//        if(do_mech)
//            M_Substep_GPU();
        HM_Substep_GPU();
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
        //SaveDAT_GPU(it);
    }

    cudaFree(dev_Pw);
    cudaFree(dev_Sw);
    cudaFree(dev_Pw_old);
    cudaFree(dev_Sw_old);
    cudaFree(dev_qx);
    cudaFree(dev_qy);
    cudaFree(dev_Krx);
    cudaFree(dev_Kry);
    cudaFree(dev_rsd_h);

    cudaFree(dev_Txx);
    cudaFree(dev_Tyy);
    cudaFree(dev_Txy);
    cudaFree(dev_Ux);
    cudaFree(dev_Uy);
    cudaFree(dev_Ux_old);
    cudaFree(dev_Uy_old);
    cudaFree(dev_Vx);
    cudaFree(dev_Vy);
    cudaFree(dev_rsd_m_x);
    cudaFree(dev_rsd_m_y);


    cudaEventRecord(tend);
    cudaEventSynchronize(tend);

    float comptime = 0.0;
    cudaEventElapsedTime(&comptime, tbeg, tend);
    printf("\nComputation time = %f s\n", comptime/1e3);

    delete [] Pw;
    delete [] Sw;
    delete [] qx;
    delete [] qy;
    delete [] rsd_h;

    delete [] Tyy;
    delete [] Txx;
    delete [] Txy;
    delete [] Ux;
    delete [] Uy;
    delete [] rsd_m_x;
    delete [] rsd_m_y;
}


__global__ void kernel_SetIC(DAT *Txx, DAT *Tyy, DAT *Txy,
                             DAT *Vx, DAT *Vy, DAT *Ux, DAT *Uy,
                             DAT *rsd_m_x, DAT *rsd_m_y,
                             DAT *Pw, DAT *Sw, DAT *qx, DAT *qy, DAT *Krx, DAT *Kry, DAT *rsd_h,
                             const DAT rhow, const DAT vg_a, const DAT vg_m, const DAT vg_n,
                             const int nx, const int ny, const DAT Lx, const DAT Ly)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    const DAT dx = Lx/nx, dy = dx;

    DAT x = (i+0.5)*dx, y = (j+0.5)*dy;
    // Cell variables
    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+j*nx;

        if(sqrt((Lx/2.0-x)*(Lx/2.0-x) + (Ly/2.0-y)*(Ly/2.0-y)) < 0.004)
            Sw[ind] = 0.1; //Pw[i+j*nx] = 1e3;
        else
            Sw[ind] = 0.04;//-1e5;

        DAT Pcc = rhow*9.81/vg_a * pow(pow(Sw[ind],-1./vg_m) - 1., 1./vg_n);
        Pw[ind] = -Pcc;
        rsd_h[ind] = 0.0;

        Txx[ind] = 0.0;
        Tyy[ind] = 0.0;
        rsd_m_x[i+j*nx] = 0.0;
        rsd_m_y[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >=0 && i <= nx && j >=0 && j < ny){
        int ind = i+j*(nx+1);
        qx[ind] = 0.0;
        Krx[ind] = 1.0;
        Ux[ind] = 0.0;
        Vx[ind] = 0.0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >=0 && i < nx && j >=0 && j <= ny){
        int ind = i+j*nx;
        qy[ind] = 0.0;
        Kry[ind] = 1.0;
        Vy[ind] = 0.0;
        Uy[ind] = 0.0;
//        if(j > 0 && j < ny)
//            Vy[ind] = 1e-3;
    }

    if(i >= 0 && i <= nx && j >= 0 && j <= ny){
        Txy[i+j*(nx+1)] = 0.0;
    }
}

__global__ void kernel_Update_V(DAT *Vx, DAT *Vy, DAT *Txx, DAT *Tyy, DAT *Txy, DAT *Pw, DAT *Sw,
                                const int nx, const int ny, const DAT dx, const DAT dy,
                                const DAT rho_s, const DAT g, const DAT dt_m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        DAT dTxxdx  = (Txx[i+j*nx] - Txx[i-1+j*nx]) / dx;
        DAT dPwSwdx = (Sw[i+j*nx]*Pw[i+j*nx] - Sw[i-1+j*nx]*Pw[i-1+j*nx]) / dx;

        Vx[i+j*(nx+1)]     += dt_m * (1./rho_s*(dTxxdx - dPwSwdx));

        //if(j > 0 && j < ny-1)
            Vx[i+j*(nx+1)] += dt_m/rho_s * (Txy[i+(j+1)*(nx+1)] - Txy[i+j*(nx+1)]) / dy;
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        DAT dTyydy = (Tyy[i+j*nx] - Tyy[i+(j-1)*nx]) / dy;
        DAT dPwSwdy = (Sw[i+j*nx]*Pw[i+j*nx] - Sw[i+(j-1)*nx]*Pw[i+(j-1)*nx]) / dy;

        Vy[i+j*nx]     += dt_m * (1./rho_s*(dTyydy - dPwSwdy) - g);

        //if(i > 0 && i < nx-1)
            Vy[i+j*nx] += dt_m/rho_s * (Txy[i+1+j*(nx+1)] - Txy[i+j*(nx+1)]) / dx;
    }

    // BC
    if(i == 0 && j >= 0 && j < ny){ // Left BCs: zero stress
        //Vx[i+j*(nx+1)] += dt_m * (1./rho_s*(Txx[i+j*nx]-0.)/dx);
    }
    if(i == nx && j >= 0 && j < ny){ // Right BCs: zero stress
        //Vx[i+j*(nx+1)] += dt_m/rho_s * (0.-Txx[i-1+j*nx])/dx;
    }
    if(j == 0 && i >= 0 && i < nx){ // Lower BCs: stress equal to water pressure?
        //Vy[i+j*nx] += dt_m * (1./rho_s*(Tyy[i+0*nx]-Pw[i+0*nx])/dy - g);
    }
}

__global__ void kernel_Update_U(DAT *Ux, DAT *Uy, DAT *Vx, DAT *Vy,
                                const int nx, const int ny, const DAT dt_m)

{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i <= nx && j >= 0 && j <= ny-1){
        Vx[i+j*(nx+1)] /= (1. + 1e-1/nx);
        Ux[i+j*(nx+1)] += dt_m * Vx[i+j*(nx+1)];
    }

    if(i >= 0 && i <= nx-1 && j >= 0 && j <= ny){
        Vy[i+j*nx] /= (1. + 1e-1/ny);
        Uy[i+j*nx] += dt_m * Vy[i+j*nx];
    }
}

__global__ void kernel_Update_Stress(DAT *Txx, DAT *Tyy, DAT *Txy, DAT *Vx, DAT *Vy,
                                     DAT *Pw, DAT *Sw, DAT *Sw_old,
                                     DAT *rsd_m_x, DAT *rsd_m_y,
                                     const int nx, const int ny,
                                     const DAT dx, const DAT dy, const DAT dt,
                                     const DAT rho_s, const DAT g, const DAT mu, const DAT lam,
                                     const DAT dt_m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    // Update Txy first
    if(i > 0 && i < nx && j > 0 && j < ny){ // shit?
        DAT dVxdy, dVydx;
        dVxdy = dVydx = 0.0;
        if(j < ny)
            dVxdy = (Vx[i+j*(nx+1)] - Vx[i+(j-1)*(nx+1)])/dy;
        if(i < nx)
            dVydx = (Vy[i+j*nx]   - Vy[i-1+j*nx])/dx;
        Txy[i+j*(nx+1)] += dt_m * mu*(dVxdy + dVydx);
    }

    // Update diagonal stress components
    if(i >= 0 && i < nx && j >= 0 && j < ny){
    //if(i >= 0 && i < nx && j > 0 && j < ny-1){    // shit
        int ind = i+nx*j;

        DAT dVxdx = (Vx[i+1+j*(nx+1)] - Vx[i+j*(nx+1)])/dx;
        DAT dVydy = (Vy[i+(j+1)*nx]   - Vy[i+j*nx])/dy;
        DAT dSwdt = (Sw[ind] - Sw_old[ind])/dt;

        Txx[ind] += dt_m * ((2*mu+lam)*(dVxdx-0.87*dSwdt-0e-2*Sw[ind]+0.0) + lam*dVydy);
        Tyy[ind] += dt_m * ((2*mu+lam)*(dVydy-0.87*dSwdt-0e-2*Sw[ind]+0.0) + lam*dVxdx);
    }

    // Residual x-component, 'i' is x-index of a vertical face
    if(i >= 1 && i <= nx-1 && j > 0 && j < ny-1){
        DAT dTxxdx  = (Txx[i+j*nx] - Txx[i-1+j*nx]) / dx;
        DAT dTxydy  = (Txy[i+(j+1)*(nx+1)] - Txy[i+j*(nx+1)]) / dy;
        DAT dPwSwdx = (Pw[i+j*nx]*Sw[i+j*nx] - Pw[i-1+j*nx]*Sw[i-1+j*nx]) / dx;
        rsd_m_x[i+j*nx] = fabs(dTxxdx + dTxydy - dPwSwdx);
//        rsd_m_x[i+j*nx] = fabs(Vx[i+j*(nx+1)]);
    }
    // Residual y-component, 'j' is y-index of a horizontal face
    if(j >= 1 && j <= ny-1 && i > 0 && i < nx-1){
        DAT dTyydy  = (Tyy[i+j*nx] - Tyy[i+(j-1)*nx]) / dy;
        DAT dTxydx  = (Txy[i+1+j*(nx+1)] - Txy[i+j*(nx+1)]) / dx;
        DAT dPwSwdy = (Pw[i+j*nx]*Sw[i+j*nx] - Pw[i+(j-1)*nx]*Sw[i+(j-1)*nx]) / dx;
        rsd_m_y[i+j*nx] = fabs(dTyydy + dTxydx - dPwSwdy - rho_s*g);
//        rsd_m_y[i+j*nx] = fabs(Vy[i+j*nx]);
    }

}


__global__ void kernel_Compute_Sw(DAT *Pw, DAT *Sw,
                             const int nx, const int ny,
                             const DAT rhow, const DAT g,
                             const DAT vg_a, const DAT vg_m, const DAT vg_n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    //printf("SHOULDN't BE HERE!!!!!!!!!!!!\n");

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        if(Pw[i+j*nx] >= 0.0)
            Sw[i+j*nx] = 1.0;
        else{
            Sw[i+j*nx] = pow(1.0 + pow(-vg_a/rhow/g*Pw[i+j*nx], vg_n), -vg_m);
        }
    }
}

__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pw, DAT *Krx, DAT *Kry,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT K,  const DAT rhow, const DAT muw, const DAT g)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // qx: (nx+1)xny
    // in matlab: qx(2:end-1,:)
    // 2:nx,:
    // 1:nx-1,:
    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal fluxes
        qx[i+j*(nx+1)] = -rhow*K/muw*Krx[i+j*(nx+1)]*((Pw[i+j*nx] - Pw[i-1+j*nx])/dx);
        //if(i==1 && j==0)
        //    printf("qx at cell 0 = %lf\n",qx[i+j*(nx+1)]);
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal fluxes
        qy[i+j*nx] = -rhow*K/muw*Kry[i+j*nx]*((Pw[i+j*nx] - Pw[i+(j-1)*nx])/dy + rhow*g);
    }

    // Bc at lower side
    if(j == 0){
        DAT Lx = nx*dx, Ly = ny*dy;
        DAT x  =  (0.5+i)*dx,  y =  (0.5+j)*dy;
        if(x > Lx/2.-Lx/8. && x < Lx/2.+Lx/8.)
        //if(i >= 14 && i <= nx-15)
            qy[i+0*nx] = FLUX;//rhow*K/muw*((Pw[i+0*nx] - 0 +)/dy + 0*rhow*g);
        else
            qy[i+0*nx] = 0.0;
    }
    //if(j == ny)
    //    qy[i+j*nx] = 0.0;
    //if(i == 0 || i == nx)
    //    qx[i+j*(nx+1)] = 0.0;
}

__global__ void kernel_Compute_Kr(DAT *qx, DAT *qy, DAT *Pw, DAT *Sw, DAT *Krx, DAT *Kry,
                                 const int nx, const int ny, const DAT vg_m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        DAT Swupw;
        if(Pw[i-1+j*nx] > Pw[i+j*nx])
            Swupw = Sw[i-1+j*nx];
        else
            Swupw = Sw[i+j*nx];

        if(Swupw > 1.0 && Swupw < 0.0)
            printf("Bad Sw = %e\n", Swupw);

        // !!!! CENTRAL
        //Swupw = 0.5*(Sw[i+j*nx]+Sw[i-1+j*nx]);

        Krx[i+j*(nx+1)] = sqrt(Swupw) * pow(pow(1.-pow(Swupw,1./vg_m),vg_m)-1.,2.);
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        DAT Swupw;
        if(Pw[i+(j-1)*nx] > Pw[i+j*nx]) // Todo: upwind based on head rather than pressure
            Swupw = Sw[i+(j-1)*nx];
        else
            Swupw = Sw[i+j*nx];
        if(isinf(Swupw) || isnan(Swupw)){
            printf("Bad Sw\n");
        }

        // !!!! CENTRAL
        //Swupw = 0.5*(Sw[i+j*nx]+Sw[i+(j-1)*nx]);

        if(Swupw > 1.0 && Swupw < 0.0)
            printf("Bad Sw = %e\n", Swupw);

        Kry[i+j*nx] = sqrt(Swupw) * pow(pow(1.-pow(Swupw,1./vg_m),vg_m)-1.,2.);
        if(isinf(Kry[i+j*nx]) || isnan(Kry[i+j*nx])){
            printf("Bad Kr\n");
        }
    }
}

__global__ void kernel_Update_Pw(DAT *rsd, DAT *Pw, DAT *Sw, DAT *Pw_old, DAT *Sw_old,
                                 DAT *qx, DAT *qy,
                                 DAT *Ux, DAT *Uy, DAT *Ux_old, DAT *Uy_old,
                                 const int nx, const int ny,
                                 const DAT dx,  const DAT dy,   const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    //const DAT dt_h = 1e-2;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        DAT divU     = (Ux[i+1+j*(nx+1)]     - Ux[i+j*(nx+1)])/dx
                     + (Uy[i+(j+1)*nx] -       Uy[i+j*nx])/dy;
        DAT divU_old = (Ux_old[i+1+j*(nx+1)] - Ux_old[i+j*(nx+1)])/dx
                     + (Uy_old[i+(j+1)*nx]    - Uy_old[i+j*nx])/dy;

        rsd[ind] = phi*rhow * (Sw[ind] - Sw_old[ind])/dt
                 + rhow*sstor * Sw[ind] * (Pw[ind] - Pw_old[ind])/dt
                 + (qx[i+1+j*(nx+1)] - qx[i+j*(nx+1)])/dx
                 + (qy[i+(j+1)*nx] - qy[i+j*nx])/dy
                 + (divU - divU_old)/dt;

        Pw[ind]  -= rsd[ind] * dt_h;

        rsd[ind] = fabs(rsd[ind]);

        //if(i==nx-1 && j==ny-2 && fabs(rsd[ind])>1e-18)
        //    printf("rsd = %lf\n", rsd[ind]);
        //    Pw[ind] = 1e11;
        //Pw[ind] = 1e11;
    }
}

__global__ void kernel_Update_Sw(DAT *rsd, DAT *Pw, DAT *Sw, DAT *Pw_old, DAT *Sw_old,
                                 DAT *qx, DAT *qy,
                                 DAT *Ux, DAT *Uy, DAT *Ux_old, DAT *Uy_old,
                                 const int nx, const int ny,
                                 const DAT dx,  const DAT dy,   const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor,
                                 const DAT muw, const DAT K,
                                 const DAT vg_a, const DAT vg_m, const DAT vg_n, const DAT DTAU)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;

        DAT divU     = (Ux[i+1+j*(nx+1)]     - Ux[i+j*(nx+1)])/dx
                     + (Uy[i+(j+1)*nx] -       Uy[i+j*nx])/dy;
        DAT divU_old = (Ux_old[i+1+j*(nx+1)] - Ux_old[i+j*(nx+1)])/dx
                     + (Uy_old[i+(j+1)*nx]   - Uy_old[i+j*nx])/dy;
        divU = divU_old = 0;

        // Compute residual
        DAT div_q  = (qx[i+1+j*(nx+1)] - qx[i+j*(nx+1)])/dx
                   + (qy[i+(j+1)*nx]   - qy[i+j*nx])/dy;

        DAT v = 1.0;

        rsd[ind]   = phi*rhow * (Sw[ind] - Sw_old[ind])/dt + div_q + v*phi*rhow*(divU-divU_old)/dt;

        // Determine pseudo-transient step
        DAT D;    // "Diffusion coefficient" for the cell
        DAT dtau; // Pseudo-transient step

        D    = rhow * K / muw / phi;
        dtau = min(dx*dx,dy*dy) / D / 4.1; // Greater D - smaller  dtau
        dtau *= 0.1;

        dtau = DTAU;

        bool use_1 = true;

        if(use_1){
            DAT Swp = Sw[ind];

            Sw[ind] = Sw[ind]*(v*(divU-divU_old)/dt + 1./dtau) + Sw_old[ind]/dt - div_q/phi/rhow;
            Sw[ind] = Sw[ind]/(1./dt + 1./dtau);


            //Sw[ind] = Swp - 1e-3*dtau * rsd[ind];

            if(Sw[ind] > 1.0 || Sw[ind] < 0.0 || isinf(Sw[ind]) || isnan(Sw[ind])){
                if(i == 30 && j == 36)
                printf("Bad Sw = %e at cell (%d %d), Swp = %e, Sw_old = %e, divq = %e, dtau = %e\n", Sw[ind], i, j, Swp, Sw_old[ind], div_q/phi/rhow, dtau);
                //Sw[ind] = Swp;
                //__trap();
                return;
            }
        }
        else{

        }
        DAT Pcc = rhow*9.81/vg_a * pow(pow(Sw[ind],-1./vg_m) - 1., 1./vg_n); // (S^(-1/m)-1)^(1/n)
        if(isnan(Pcc) || isinf(Pcc)){
            printf("Bad Pcc at cell (%d %d), Sw = %e\n", i, j, Sw[ind]);
            __trap();
        }
        Pw[ind] = -Pcc;

        rsd[ind]   = fabs(rsd[ind]);
    }
}

void Problem::SaveVTK_GPU(std::string path)
{
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pw, dev_Pw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sw, dev_Sw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    cudaMemcpy(Tyy, dev_Tyy, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Txx, dev_Txx, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Txy, dev_Txy, sizeof(DAT) * (nx+1)*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ux, dev_Ux, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Uy, dev_Uy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Vx, dev_Vx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vy, dev_Vy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    cudaMemcpy(rsd_m_x, dev_rsd_m_x, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(rsd_m_y, dev_rsd_m_y, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(rsd_h, dev_rsd_h, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);

    SaveVTK(path);
}

void Problem::SaveDAT_GPU(int stepnum)
{
    cudaMemcpy(Pw, dev_Pw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sw, dev_Sw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    cudaMemcpy(Tyy, dev_Tyy, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Txx, dev_Txx, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ux, dev_Ux, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Uy, dev_Uy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Vx, dev_Vx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vy, dev_Vy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    std::string path = "C:\\Users\\Denis\\Documents\\msu_thmc\\MATLAB\\res_gpu\\swell\\";
    FILE *f;
    std::string fname;

    fname = path + "Ux" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Ux, sizeof(DAT), (nx+1)*ny, f);
    fclose(f);

    fname = path + "Uy" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Uy, sizeof(DAT), (ny+1)*nx, f);
    fclose(f);

    fname = path + "Sw" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Sw, sizeof(DAT), ny*nx, f);
    fclose(f);

    fname = path + "Pw" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Pw, sizeof(DAT), ny*nx, f);
    fclose(f);
}
