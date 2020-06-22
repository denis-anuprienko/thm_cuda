#include "header.h"
#include <cuda.h>

using namespace std;

#define BLOCK_DIM 16

__global__ void kernel_SetIC(DAT *Pw, DAT *Sw, DAT *qx, DAT *qy, DAT *Krx, DAT *Kry, DAT *rsd,
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
                                 DAT *qx, DAT *qy, const int nx, const int ny,
                                 const DAT dx, const DAT dy, const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor);

void Problem::SetIC_GPU()
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((nx+dimBlock.x-1)/dimBlock.x, (ny+dimBlock.y-1)/dimBlock.y);
    printf("Launching %dx%d blocks of %dx%d threads\n", (nx+1+dimBlock.x-1)/dimBlock.x,
           (ny+1+dimBlock.y-1)/dimBlock.y, BLOCK_DIM, BLOCK_DIM);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    kernel_SetIC<<<dimGrid,dimBlock>>>(dev_Pw, dev_Sw, dev_qx, dev_qy, dev_Krx, dev_Kry, dev_rsd_h,
                                       nx, ny, Lx, Ly);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at SetIC\n", err);
    //kernel_Compute_Sw<<<dimGrid,dimBlock>>>(Pw, Sw, rhow, g, vg_a, vg_m, vg_n);
    //kernel_ComputeKr<<<dimGrid,dimBlock>>>(H, Theta, Krx, Kry, nx, ny, dy);
    //kernel_ComputeFluidFluxes<<<dimGrid,dimBlock>>>(H, qx, qy, Krx, Kry, nx, ny, dx, dy, D);
    Compute_Sw_GPU();
    //Compute_Kr_GPU();
    //Compute_Q_GPU();
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
                                           dev_qx, dev_qy, nx, ny, dx, dy, dt,
                                           phi, rhow, sstor);
    cudaError_t err = cudaGetLastError();
    if(err != 0)
        printf("Error %x at Pw\n", err);
}

void Problem::SolveOnGPU()
{
    cudaMalloc((void**)&dev_Pw,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sw,     sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Pw_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_Sw_old, sizeof(DAT) * nx*ny);
    cudaMalloc((void**)&dev_qx,     sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_qy,     sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_Krx,    sizeof(DAT) * (nx+1)*ny);
    cudaMalloc((void**)&dev_Kry,    sizeof(DAT) * nx*(ny+1));
    cudaMalloc((void**)&dev_rsd_h,     sizeof(DAT) * nx*ny);

//    std::fill(dev_Pw, dev_Pw + nx*ny, 0.0);
//    std::fill(dev_Pw, dev_Sw + nx*ny, 0.0);
//    std::fill(dev_Pw, dev_qx + (nx+1)*ny, 0.0);
//    std::fill(dev_Pw, dev_qy + nx*(ny+1), 0.0);
//    std::fill(dev_Pw, dev_Krx + (nx+1)*ny, 0.0);
//    std::fill(dev_Pw, dev_Kry + nx*(ny+1), 0.0);
//    std::fill(dev_Pw, dev_rsd_h + nx*ny, 0.0);

    // Still needed for VTK saving
    Pw    = new DAT[nx*ny];
    Sw    = new DAT[nx*ny];
    qx    = new DAT[(nx+1)*ny];
    qy    = new DAT[nx*(ny+1)];
    rsd_h = new DAT[nx*ny];

    SetIC_GPU();
    cudaDeviceSynchronize();
    SaveVTK_GPU(respath + "/sol0.vtk");

    for(int it = 1; it <= nt; it++){
        printf("\n\n =======  TIME = %lf s =======\n", it*dt);
        cudaMemcpy(dev_Pw_old, dev_Pw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_Sw_old, dev_Sw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToDevice);
        for(int nit = 1; nit < niter*1; nit++){
            Compute_Sw_GPU();
            Compute_Kr_GPU();
            Compute_Q_GPU();
            Update_Pw_GPU();
            if(nit%10000 == 0 || nit == 1){
                cudaMemcpy(rsd_h, dev_rsd_h, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
                DAT err = 0;
                for(int i = 0; i < nx*ny; i++){
                    if(fabs(rsd_h[i]) > err)
                        err = fabs(rsd_h[i]);
                    if(isinf(rsd_h[i]) || isnan(rsd_h[i])){
                        printf("Bad value, iter %d", nit);
                        exit(0);
                    }
                }
                printf("iter %d: r_w = %e\n", nit, err);
                fflush(stdout);
                if(err < eps_a_h){
                    printf("Flow converged in %d it.: r_w = %e\n", nit, err);
                    break;
                }
            }
        }
        string name = respath + "/sol" + to_string(it) + ".vtk";
        SaveVTK_GPU(name);
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

    delete [] Pw;
    delete [] Sw;
    delete [] qx;
    delete [] qy;
    delete [] rsd_h;
}


__global__ void kernel_SetIC(DAT *Pw, DAT *Sw, DAT *qx, DAT *qy, DAT *Krx, DAT *Kry, DAT *rsd,
                             const int nx, const int ny, const DAT Lx, const DAT Ly)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    const DAT dx = Lx/nx, dy = dx;

    DAT x = (i+0.5)*dx, y = (j+0.5)*dy;
    // Cell variables
    if(i >= 0 && i < nx && j >= 0 && j < ny){
        //if(i*i + j*j < 400)
        //if(sqrt((Lx/2.0-x)*(Lx/2.0-x) + (Ly/2.0-y)*(Ly/2.0-y)) < 0.001)
        //    Pw[i+j*nx] = 1e3;
        //else
        //    Pw[i+j*nx] = -1e5;
        //DAT rad = (DAT)(i*i + j*j);
        //Pw[i+j*nx] = sqrt(rad);
        Pw[i+j*nx] = -1e5;
        rsd[i+j*nx] = 0.0;
    }
    // Vertical face variables - x-fluxes, for example
    if(i >=0 && i <= nx && j >=0 && j < ny){
        int ind = i+j*(nx+1);
        qx[ind] = 0.0;
        Krx[ind] = 1.0;
    }
    // Horizontal face variables - y-fluxes, for example
    if(i >=0 && i < nx && j >=0 && j <= ny){
        int ind = i+j*nx;
        qy[ind] = 0.0;
        Kry[ind] = 1.0;
    }
}

__global__ void kernel_Compute_Sw(DAT *Pw, DAT *Sw,
                             const int nx, const int ny,
                             const DAT rhow, const DAT g,
                             const DAT vg_a, const DAT vg_m, const DAT vg_n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

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
        DAT x  =  i*dx,  y =  j*dy;
        //if(x > Lx/2.-Lx/8. && x < Lx/2.+Lx/8.)
        if(i >= 14 && i <= nx-15)
            qy[i+0*nx] = -rhow*K/muw*((Pw[i+0*nx] - 1e3)/dy + rhow*g);
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

        // !!!! CENTRAL
        Swupw = 0.5*(Sw[i+j*nx]+Sw[i-1+j*nx]);

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
        Swupw = 0.5*(Sw[i+j*nx]+Sw[i+(j-1)*nx]);

        Kry[i+j*nx] = sqrt(Swupw) * pow(pow(1.-pow(Swupw,1./vg_m),vg_m)-1.,2.);
        if(isinf(Kry[i+j*nx]) || isnan(Kry[i+j*nx])){
            printf("Bad Kr\n");
        }
    }
}

__global__ void kernel_Update_Pw(DAT *rsd, DAT *Pw, DAT *Sw, DAT *Pw_old, DAT *Sw_old,
                                 DAT *qx, DAT *qy, const int nx, const int ny,
                                 const DAT dx,  const DAT dy,   const DAT dt,
                                 const DAT phi, const DAT rhow, const DAT sstor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    const DAT dt_h = 1e-3;

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        int ind = i+nx*j;
        rsd[ind] = phi*rhow * (Sw[ind] - Sw_old[ind])/dt
                 + rhow*sstor * Sw[ind] * (Pw[ind] - Pw_old[ind])/dt
                 + (qx[i+1+j*(nx+1)] - qx[i+j*(nx+1)])/dx
                 + (qy[i+(j+1)*nx] - qy[i+j*nx])/dy;

        Pw[ind]  -= rsd[ind] * dt_h;
        //if(i==nx-1 && j==ny-2 && fabs(rsd[ind])>1e-18)
        //    printf("rsd = %lf\n", rsd[ind]);
        //    Pw[ind] = 1e11;
        //Pw[ind] = 1e11;
    }
}

void Problem::SaveVTK_GPU(std::string path)
{
    // Copy data from device and perform standard SaveVTK

    cudaMemcpy(Pw, dev_Pw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sw, dev_Sw, sizeof(DAT) * nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qx, dev_qx, sizeof(DAT) * (nx+1)*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, dev_qy, sizeof(DAT) * nx*(ny+1), cudaMemcpyDeviceToHost);

    SaveVTK(path);
}
