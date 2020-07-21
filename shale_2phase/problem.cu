#include "header.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define BLOCK_DIM 16
#define dt_h      5e7
#define dt_m      1e-6

void FindMax(DAT *dev_arr, DAT *max, int size);

__global__ void kernel_SetIC(DAT *Pl, DAT *Pg, DAT *Sl,
                             DAT *Kx, DAT *Ky,
                             DAT *Krlx, DAT *Krly, DAT *Krgx, DAT *Krgy,
                             DAT *qlx, DAT *qly, DAT *qgx, DAT *qgy,
                             DAT *phi,
                             DAT *rsd_l, DAT *rsd_g,
                             const DAT K0,
                             const int nx, const int ny, const DAT Lx, const DAT Ly);

__global__ void kernel_Compute_Q();

__global__ void kernel_Compute_K();

__global__ void kernel_Update_P();

__global__ void kernel_Update_Poro();

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
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Pl, dev_Pg, dev_Sl,
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
}


void Problem::Compute_Q_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_Q<<<dimGrid,dimBlock>>>();
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Q\n", err);
}

void Problem::Compute_K_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
    kernel_Compute_K<<<dimGrid,dimBlock>>>();
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at K\n", err);
}

void Problem::Compute_Kr_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
}

void Problem::Compute_S_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+1+dimBlock.x-1)/dimBlock.x, (ny+1+dimBlock.y-1)/dimBlock.y);
}

void Problem::Update_P_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_P<<<dimGrid,dimBlock>>>();
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Pf\n", err);
}

void Problem::Update_Poro()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    kernel_Update_Poro<<<dimGrid,dimBlock>>>();
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Poro\n", err);
}

void Problem::H_Substep_GPU()
{
    printf("Flow\n");
    fflush(stdout);
    DAT err_l = 1, err_g = 1, err_l_old, err_g_old;
    for(int nit = 1; nit <= niter; nit++){
        Compute_K_GPU();
        Compute_S_GPU();
        Compute_Kr_GPU();
        Compute_Q_GPU();
        Update_P_GPU();
        if(nit%10000 == 0 || nit == 1){
            err_l_old = err_l;
            err_g_old = err_g;
            FindMax(dev_rsd_l, &err_l, nx*ny);
            FindMax(dev_rsd_g, &err_g, nx*ny);
            printf("iter %d: r_l = %e, r_g = %e\n", nit, err_l, err_g);
            fflush(stdout);
            if((    (err_l<eps_a_h && err_g<eps_a_h) ||
                    (fabs(err_l-err_l_old) < 1e-15 && fabs(err_g-err_g_old) < 1e-15))
                && nit > 10000){
                printf("Flow converged in %d it.: r_l = %e, r_g = %e\n", nit, err_l, err_g);
                break;
            }
        }
    }
    //Update_Poro();
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
    cudaMalloc((void**)&dev_rsd_l,  sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_rsd_g,  sizeof(DAT) * nx*ny);
    cudaEventRecord(tbeg);

    printf("Allocated on GPU\n");

    // Still needed for VTK saving
    Pl      = new DAT[nx*ny];
    qlx      = new DAT[(nx+1)*ny];
    qly      = new DAT[nx*(ny+1)];
    Kx      = new DAT[(nx+1)*ny];
    Ky      = new DAT[nx*(ny+1)];
    phi     = new DAT[nx*ny];

    SetIC_GPU();
    cudaDeviceSynchronize();

    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME STEP %d, T = %lf s =======\n", it, it*dt);
        if(do_mech)
;//            M_Substep_GPU();
        cudaMemcpy(dev_Pl_old, dev_Pl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Pg_old, dev_Pg, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Sl_old, dev_Sl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        H_Substep_GPU();
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
        SaveDAT_GPU(it);
    }

    cudaFree(dev_Pl);
    cudaFree(dev_Pl_old);
    cudaFree(dev_Pg);
    cudaFree(dev_Pg_old);
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
    cudaFree(dev_rsd_l);
    cudaFree(dev_rsd_g);

    cudaEventRecord(tend);
    cudaEventSynchronize(tend);

    float comptime = 0.0;
    cudaEventElapsedTime(&comptime, tbeg, tend);
    printf("\nComputation time = %f s\n", comptime/1e3);

    delete [] Pl;
    delete [] qlx;
    delete [] qly;
    delete [] Kx;
    delete [] Ky;
    delete [] phi;
}


__global__ void kernel_SetIC(DAT *Pl, DAT *Pg, DAT *Sl,
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
            Pl[i+j*nx] = 10e6;
        else
            Pl[i+j*nx] = 8e6;

        Pg[i+j*nx] = 8e6;
        phi[i+j*nx] = 0.16;
        rsd_l[i+j*nx] = 0.0;
        rsd_g[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >= 0 && i <= nx && j >= 0 && j < ny){
        int ind = i+j*(nx+1);
        qlx[ind] = 0.0;
        qgx[ind] = 0.0;
        Kx[ind] = K0;
        Krlx[ind] = 1.0;
        Krgx[ind] = 1.0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >= 0 && i < nx && j >= 0 && j <= ny){
        int ind = i+j*nx;
        qly[ind] = 0.0;
        qgy[ind] = 0.0;
        Ky[ind] = K0;
        Krly[ind] = 1.0;
        Krgy[ind] = 1.0;
    }
}



__global__ void kernel_Compute_Q()
{

}

__global__ void kernel_Compute_K()
{

}

__global__ void kernel_Update_P()
{

}

__global__ void kernel_Update_Poro()
{

}

void Problem::SaveVTK_GPU(std::string path)
{
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pl,  dev_Pl, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qlx, dev_qlx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qly, dev_qly, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kx,  dev_Kx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ky,  dev_Ky, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(phi, dev_phi, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);

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
