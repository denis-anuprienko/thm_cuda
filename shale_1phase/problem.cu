#include "header.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define BLOCK_DIM 16
#define dt_h      1e-2
#define dt_m      1e-6

void FindMax(DAT *dev_arr, DAT *max, int size);

__global__ void kernel_SetIC(DAT *Pf, DAT *qx, DAT *qy, DAT *Kx, DAT *Ky,
                             DAT *rsd_h,
                             const int nx, const int ny, const DAT Lx, const DAT Ly);

__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT rhow, const DAT muw, const DAT g);
__global__ void kernel_Compute_K(DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT vg_m);

__global__ void kernel_Update_Pf(DAT *rsd, DAT *Pf, DAT *Pf_old,
                                 DAT *qx, DAT *qy, const int nx, const int ny,
                                 const DAT dx, const DAT dy, const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor);

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
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Pf, dev_qx, dev_qy, dev_Kx, dev_Ky, dev_rsd_h,
                                       nx, ny, Lx, Ly);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at SetIC\n", err);
}


void Problem::Compute_Q_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Q<<<dimGrid,dimBlock>>>(dev_qx, dev_qy, dev_Pf, dev_Kx, dev_Ky,
                                           nx, ny, dx, dy, rhow, muw, g);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Q\n", err);
}

void Problem::Compute_K_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_K<<<dimGrid,dimBlock>>>(dev_Pf, dev_Kx, dev_Ky,
                                           nx, ny, 0.0);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Kr\n", err);
}

void Problem::Update_Pf_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Pf<<<dimGrid,dimBlock>>>(dev_rsd_h, dev_Pf, dev_Pf_old,
                                           dev_qx, dev_qy,
                                           nx, ny, dx, dy, dt,
                                           phi, rhow, 0.0);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Pw\n", err);
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
    cudaMalloc((void**)&dev_rsd_h,  sizeof(DAT) * nx*ny);
    cudaEventRecord(tbeg);

    printf("Allocated on GPU\n");

    // Still needed for VTK saving
    Pf      = new DAT[nx*ny];
    qx      = new DAT[(nx+1)*ny];
    qy      = new DAT[nx*(ny+1)];
    rsd_h   = new DAT[nx*ny];

    SetIC_GPU();
    cudaDeviceSynchronize();

    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME = %lf s =======\n", it*dt);
        if(do_mech)
;//            M_Substep_GPU();
        cudaMemcpy(dev_Pf_old, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        if(do_flow)
;//            H_Substep_GPU();
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
        SaveDAT_GPU(it);
    }

    cudaFree(dev_Pf);
    cudaFree(dev_Pf_old);
    cudaFree(dev_qx);
    cudaFree(dev_qy);
    cudaFree(dev_Kx);
    cudaFree(dev_Ky);
    cudaFree(dev_rsd_h);

    cudaEventRecord(tend);
    cudaEventSynchronize(tend);

    float comptime = 0.0;
    cudaEventElapsedTime(&comptime, tbeg, tend);
    printf("\nComputation time = %f s\n", comptime/1e3);

    delete [] Pf;
    delete [] qx;
    delete [] qy;
    delete [] rsd_h;
}


__global__ void kernel_SetIC(DAT *Pf, DAT *qx, DAT *qy, DAT *Kx, DAT *Ky, DAT *rsd_h,
                             const int nx, const int ny, const DAT Lx, const DAT Ly)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    const DAT dx = Lx/nx, dy = dx;

    DAT x = (i+0.5)*dx, y = (j+0.5)*dy;
    // Cell variables
    if(i >= 0 && i < nx && j >= 0 && j < ny){
        //if(i*i + j*j < 400)
//        if(sqrt((Lx/2.0-x)*(Lx/2.0-x) + (Ly/2.0-y)*(Ly/2.0-y)) < 0.001)
//            Pf[i+j*nx] = 1e3;
//        else
//            Pf[i+j*nx] = -1e5;
        //DAT rad = (DAT)(i*i + j*j);
        //Pf[i+j*nx] = sqrt(rad);
        Pf[i+j*nx] = -1e5;
        rsd_h[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >= 0 && i <= nx && j >=0 && j < ny){
        int ind = i+j*(nx+1);
        qx[ind] = 0.0;
        Kx[ind] = 1.0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >= 0 && i < nx && j >=0 && j <= ny){
        int ind = i+j*nx;
        qy[ind] = 0.0;
        Ky[ind] = 1.0;
    }
}



__global__ void kernel_Compute_Q(DAT *qx, DAT *qy, DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT dx, const DAT dy,
                                 const DAT rhow, const DAT muw, const DAT g)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // qx: (nx+1)xny
    // in matlab: qx(2:end-1,:)
    // 2:nx,:
    // 1:nx-1,:
    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal fluxes
        qx[i+j*(nx+1)] = -rhow/muw*Kx[i+j*(nx+1)]*((Pf[i+j*nx] - Pf[i-1+j*nx])/dx);
        //if(i==1 && j==0)
        //    printf("qx at cell 0 = %lf\n",qx[i+j*(nx+1)]);
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal fluxes
        qy[i+j*nx] = -rhow/muw*Ky[i+j*nx]*((Pf[i+j*nx] - Pf[i+(j-1)*nx])/dy + rhow*g);
    }

    // Bc at lower side
    if(j == 0){
        // todo
    }
}

__global__ void kernel_Compute_K(DAT *Pf, DAT *Kx, DAT *Ky,
                                 const int nx, const int ny, const DAT vg_m)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i > 0 && i < nx && j >= 0 && j <= ny-1){ // Internal faces
        // todo
    }

    if(i >= 0 && i <= nx-1 && j > 0 && j < ny){ // Internal faces
        // todo
    }
}

__global__ void kernel_Update_Pf(DAT *rsd, DAT *Pf, DAT *Pf_old,
                                 DAT *qx, DAT *qy,
                                 const int nx, const int ny,
                                 const DAT dx,  const DAT dy,   const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;
        // todo
    }
}

void Problem::SaveVTK_GPU(std::string path)
{
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pf, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    SaveVTK(path);
}

void Problem::SaveDAT_GPU(int stepnum)
{
    cudaMemcpy(Pf, dev_Pf, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    std::string path = "C:\\Users\\Denis\\Documents\\msu_thmc\\MATLAB\\res_gpu\\shale_1phase\\";
    FILE *f;
    std::string fname;

    fname = path + "Pf" + std::to_string(stepnum) + ".dat";
    f = fopen(fname.c_str(), "wb");
    fwrite(Pf, sizeof(DAT), ny*nx, f);
    fclose(f);
}
