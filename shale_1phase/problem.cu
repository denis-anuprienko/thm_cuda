#include "header.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define BLOCK_DIM 16
#define dt_h      5e5
#define dt_m      1e-6

void FindMax(DAT *dev_arr, DAT *max, int size);

__global__ void kernel_SetIC(DAT *Pf, DAT *qx, DAT *qy, DAT *Kx, DAT *Ky, DAT *phi,
                             DAT *rsd_h,
                             const int nx, const int ny, const DAT Lx, const DAT Ly, const DAT K0);

__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT rhof, const DAT muf, const DAT g);
__global__ void kernel_Compute_K(DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT K0,
                                 const DAT gamma, const DAT Pt, const DAT P0);

__global__ void kernel_Update_Pf(DAT *rsd, DAT *Pf, DAT *Pf_old, DAT *phi,
                                 DAT *qx, DAT *qy, const int nx, const int ny,
                                 const DAT dx, const DAT dy, const DAT dt, const DAT c_f,
                                 const DAT c_phi);

__global__ void kernel_Update_Poro(DAT *phi, DAT *Pf, DAT *Pf_old,
                                   const int nx, const int ny,
                                   const DAT dx, const DAT dy, const DAT dt,
                                   const DAT c_phi);

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

void Problem::SetIC_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    printf("Launching %dx%d blocks of %dx%d threads\n", (nx+1+dimBlock.x-1)/dimBlock.x,
           (ny+1+dimBlock.y-1)/dimBlock.y, BLOCK_DIM, BLOCK_DIM);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Pf, dev_qx, dev_qy, dev_Kx, dev_Ky,
                                       dev_phi, dev_rsd_h,
                                       nx, ny, Lx, Ly, K0);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at SetIC\n", err);
}


void Problem::Compute_Q_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Q<<<dimGrid,dimBlock>>>(dev_qx, dev_qy, dev_Pf, dev_Kx, dev_Ky,
                                           nx, ny, dx, dy, rhof, muf, g);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Q\n", err);
}

void Problem::Compute_K_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_K<<<dimGrid,dimBlock>>>(dev_Pf, dev_Kx, dev_Ky,
                                           nx, ny, K0, gamma, Pt, P0);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Kr\n", err);
}

void Problem::Update_Pf_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Pf<<<dimGrid,dimBlock>>>(dev_rsd_h, dev_Pf, dev_Pf_old, dev_phi,
                                           dev_qx, dev_qy,
                                           nx, ny, dx, dy, dt,
                                           c_f, c_phi);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Pf\n", err);
}

void Problem::Update_Poro()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Poro<<<dimGrid,dimBlock>>>(dev_phi, dev_Pf, dev_Pf_old,
                                             nx, ny,
                                             dx, dy, dt,
                                             c_phi);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Poro\n", err);
}

void Problem::H_Substep_GPU()
{
    printf("Flow\n");
    fflush(stdout);
    for(int nit = 1; nit <= niter; nit++){
        Compute_K_GPU();
        Compute_Q_GPU();
        Update_Pf_GPU();
        if(nit%10000 == 0 || nit == 1){
            DAT err = 13;
            FindMax(dev_rsd_h, &err, nx*ny);
            printf("iter %d: r_w = %e\n", nit, err);
            fflush(stdout);
            if(err < eps_a_h && nit > 10000){
                printf("Flow converged in %d it.: r_w = %e\n", nit, err);
                break;
            }
        }
    }
    Update_Poro();
    P_upstr.push_back(Pf[0]);
}

void Problem::SolveOnGPU()
{
    cudaEvent_t tbeg, tend;
    cudaEventCreate(&tbeg);
    cudaEventCreate(&tend);
    cudaEventRecord(tbeg);
    cudaMalloc((void**)&dev_Pf,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pf_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_qx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_qy,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Kx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Ky,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_phi,    sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_rsd_h,  sizeof(DAT) * nx*ny);
    cudaEventRecord(tbeg);

    printf("Allocated on GPU\n");

    // Still needed for VTK saving
    Pf      = new DAT[nx*ny];
    qx      = new DAT[(nx+1)*ny];
    qy      = new DAT[nx*(ny+1)];
    Kx      = new DAT[(nx+1)*ny];
    Ky      = new DAT[nx*(ny+1)];
    phi     = new DAT[nx*ny];
    rsd_h   = new DAT[nx*ny];

    SetIC_GPU();
    cudaDeviceSynchronize();

    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME = %lf s =======\n", it*dt);
        if(do_mech)
;//            M_Substep_GPU();
        cudaMemcpy(dev_Pf_old, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        H_Substep_GPU();
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
        //SaveDAT_GPU(it);
    }

    std::string name = respath + "/Pupstr.txt";
    ofstream outp;
    outp.open(name);
    outp << "[";
    for(int i = 0; i < P_upstr.size(); i++)
        outp << std::to_string(P_upstr[i]) << "\n";
    outp << "]";
    outp.close();

    cudaFree(dev_Pf);
    cudaFree(dev_Pf_old);
    cudaFree(dev_qx);
    cudaFree(dev_qy);
    cudaFree(dev_Kx);
    cudaFree(dev_Ky);
    cudaFree(dev_phi);
    cudaFree(dev_rsd_h);

    cudaEventRecord(tend);
    cudaEventSynchronize(tend);

    float comptime = 0.0;
    cudaEventElapsedTime(&comptime, tbeg, tend);
    printf("\nComputation time = %f s\n", comptime/1e3);

    delete [] Pf;
    delete [] qx;
    delete [] qy;
    delete [] Kx;
    delete [] Ky;
    delete [] phi;
    delete [] rsd_h;
}


__global__ void kernel_SetIC(DAT *Pf, DAT *qx, DAT *qy, DAT *Kx, DAT *Ky, DAT *phi, DAT *rsd_h,
                             const int nx, const int ny, const DAT Lx, const DAT Ly, const DAT K0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    const DAT dx = Lx/nx, dy = Ly/ny;

    DAT x = (i+0.5)*dx, y = (j+0.5)*dy;
    // Cell variables
    if(i >= 0 && i < nx && j >= 0 && j < ny){
//        if(sqrt((Lx/2.0-x)*(Lx/2.0-x) + (Ly/2.0-y)*(Ly/2.0-y)) < 0.001)
//            Pf[i+j*nx] = 10e6;
//        else
//            Pf[i+j*nx] = 8e6;

        Pf[i+j*nx] = 8e6;
        phi[i+j*nx] = 0.16;
        rsd_h[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >= 0 && i <= nx && j >= 0 && j < ny){
        int ind = i+j*(nx+1);
        qx[ind] = 0.0;
        Kx[ind] = K0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >= 0 && i < nx && j >= 0 && j <= ny){
        int ind = i+j*nx;
        qy[ind] = 0.0;
        Ky[ind] = K0;
    }
}



__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT rhof, const DAT muf, const DAT g)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // qx: (nx+1)xny
    // in matlab: qx(2:end-1,:)
    // 2:nx,:
    // 1:nx-1,:
    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal fluxes
        qx[i+j*(nx+1)] = -1.0/muf*Kx[i+j*(nx+1)]*((Pf[i+j*nx] - Pf[i-1+j*nx])/dx);
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal fluxes
        qy[i+j*nx] = -1.0/muf*Ky[i+j*nx]*((Pf[i+j*nx] - Pf[i+(j-1)*nx])/dy + 0*rhof*g);
    }

    // Bc at upper side
    if(i >= 0 && i <= nx-1 && j == ny){
        // todo: include permeability calculation for boundary
        DAT Pupw = Pf[i+(j-1)*nx];
        DAT K = 1e-18 * exp(-0.028*1e-6*(43e6-Pupw-1e5));
        if(Pupw < 11e6)
            K *= 9e-3;
        qy[i+j*nx] = -1./muf*K*((8e6 - Pf[i+(j-1)*nx])/dy + 0*rhof*g);
    }

    if(i >= 0 && i <= nx-1 && j == 0){
        // todo: include permeability calculation for boundary
        qy[i+j*nx] = 9.6e-3/60/60/700/0.012;
    }
}

__global__ void kernel_Compute_K(DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny,
                                 const DAT K0, const DAT gamma, const DAT Pt, const DAT P0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        // Upwind approach
        //DAT Pupw = max(Pf[i+j*nx], Pf[i+1+j*nx]);
        DAT Pupw = 0.5*(Pf[i+j*nx] + Pf[i+1+j*nx]);
        Kx[i+j*(nx+1)] = K0 * exp(-gamma*(Pt-Pupw-P0));
        if(Pupw < 11e6)
            Kx[i+j*(nx+1)] *= 9e-3;
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        // Upwind
        DAT Pupw = 0.5*(Pf[i+j*nx] + Pf[i+(j+1)*nx]);
        Ky[i+j*nx] = K0 * exp(-gamma*(Pt-Pupw-P0));
        if(Pupw < 11e6)
            Ky[i+j*nx] *= 9e-3;
    }
}

__global__ void kernel_Update_Pf(DAT *rsd, DAT *Pf, DAT *Pf_old, DAT *phi,
                                 DAT *qx, DAT *qy,
                                 const int nx, const int ny,
                                 const DAT dx,  const DAT dy, const DAT dt,
                                 const DAT c_f, const DAT c_phi)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;
        DAT C_t = c_f*phi[ind] + c_phi*phi[ind]/(1.0-phi[ind]);
        rsd[ind] = C_t*(Pf[ind]-Pf_old[ind])/dt
                 + (qx[i+1+j*(nx+1)] - qx[i+j*(nx+1)])/dx
                 + (qy[i+(j+1)*nx]   - qy[i+j*nx])/dy;
        Pf[ind]  -= dt_h * rsd[ind];
        rsd[ind] = fabs(rsd[ind]);
    }
}

__global__ void kernel_Update_Poro(DAT *phi, DAT *Pf, DAT *Pf_old,
                                   const int nx, const int ny,
                                   const DAT dx, const DAT dy, const DAT dt,
                                   const DAT c_phi)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;
        // Explicit update
        //phi[ind] += 1e-6*phi[ind]*(Pf[ind] - Pf_old[ind]);

        // Implicit update
        phi[ind] /= (1.0 - c_phi*(Pf[ind] - Pf_old[ind]));
    }
}

void Problem::SaveVTK_GPU(std::string path)
{
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pf, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kx, dev_Kx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ky, dev_Ky, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(phi, dev_phi, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);

    SaveVTK(path);
}

void Problem::SaveDAT_GPU(int stepnum)
{
    cudaMemcpy(Pf, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kx, dev_Kx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ky, dev_Ky, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    std::string path = "C:\\Users\\Denis\\Documents\\msu_thmc\\MATLAB\\res_gpu\\shale_1phase\\";
    FILE *f;
    std::string fname;

    fname = path + "Pf" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Pf, sizeof(DAT), ny*nx, f);
    fclose(f);
}
