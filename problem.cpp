#include "header.h"
#include <filesystem>

using namespace std;

void Problem::Init()
{
    //cout << "Save files every " << save_intensity << " step" << endl;
    respath = std::experimental::filesystem::current_path().string() + "\\res";
    printf("Init finished\n");
    fflush(stdout);
}

void Problem::SaveVTK(string path)
{
    ofstream out;
    out.open(path);
    if(!out.is_open()){
        cout << "Couldn't open file " << path << "!";
        exit(0);
    }
    out << "# vtk DataFile Version 3.0" << endl << endl;
    out << "ASCII" << endl;
    out << "DATASET STRUCTURED_GRID" << endl;
    out << "DIMENSIONS " << nx+1 << " " << ny+1 << " 1" << endl;
    out << "POINTS " << (nx+1)*(ny+1) << " DOUBLE" << endl;

//    for(int i = 0; i <= nx; i++){
//        for(int j = 0; j <= ny; j++){
//            out << j*dy << " " << i*dx << " 0.0" << endl;
//        }
//    }

//    for(int i = 0; i <= nx; i++){
//        for(int j = 0; j <= ny; j++){
    for(int j = 0; j <= ny; j++){
        for(int i = 0; i <= nx; i++){
            DAT ux_l, ux_r, uy_l, uy_u;
            ux_l = ux_r = uy_u = uy_l = 0.0;

            if(i > 0)
                ux_l = Ux[i-1+min(j,ny-1)*(nx+1)];
            else if(i == 0)
                ux_l = Ux[i+min(j,ny-1)*(nx+1)];

            if(i < nx)
                ux_r = Ux[i+1+min(j,ny-1)*(nx+1)];
            else if(i == nx)
                ux_r = Ux[i+min(j,ny-1)*(nx+1)];

            if(j > 0)
                uy_l = Uy[min(i,nx-1)+(j-1)*nx];
            else if(j == 0)
                uy_l = Uy[min(i,nx-1)+j*nx];

            if(j < ny)
                uy_u = Uy[min(i,nx-1)+(j+1)*nx];
            else if(j == ny)
                uy_u = Uy[min(i,nx-1)+j*nx];

            //out << j*dy + 0e1*(uy_l+uy_u) << " " << i*dx + 0e1*(ux_l+ux_r) << " 0.0" << endl;
            DAT ux, uy;
            ux = uy = 0.0;
//            if(j < ny)
//                ux = Ux[i+j*(nx+1)];
//            if(i < nx)
//                uy = Uy[i+j*nx];
            //out << j*dy + 0*uy << " " << i*dx + 0*ux << " 0.0" << endl;

            ux = ux_r;//0.5 * (ux_l + ux_r);
            uy = uy_u;//0.5 * (uy_l + uy_u);

            if(i == 0)
                uy = Uy[i+j*nx];
            if(i == nx)
                uy = Uy[i-1+j*nx];
            if(j == 0)
                ux = Ux[i+j*(nx+1)];
            if(j == ny)
                ux = Ux[i+(j-1)*(nx+1)];
            out << i*dx + ux << " " << j*dy + uy << " 0.0" << endl;
        }
    }

    out << "CELL_DATA " << nx * ny << endl;

    out << "SCALARS Water_Pressure double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Pw[i+j*nx] << endl;
        }
    }

    out << "SCALARS Water_Saturation double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Sw[i+j*nx] << endl;
        }
    }

    out << "SCALARS Tyy double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Tyy[i+j*nx] << endl;
        }
    }

    out << "SCALARS Txx double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Txx[i+j*nx] << endl;
        }
    }

    out << "VECTORS FluidFlux double" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << 0.5*(qx[i+j*(nx+1)]+qx[i+1+j*(nx+1)]) << " " <<
                   0.5*(qy[i+(j+1)*nx]+qy[i+j*nx]) << " 0.0" << endl;
        }
    }

    out << "VECTORS Displacement double" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            DAT ux_l, ux_r, uy_l, uy_u;
            ux_l = ux_r = uy_u = uy_l = 0.0;
            if(i > 0)
                ux_l = Ux[i-1+j*(nx+1)];
            if(i < nx)
                ux_r = Ux[i+1+j*(nx+1)];
            if(j > 0)
                uy_l = Uy[i+(j-1)*nx];
            if(j < ny)
                uy_u = Uy[i+(j+1)*nx];
            out << 0.5*(ux_l+ux_r) << " " <<
                   0.5*(uy_u+uy_l) << " 0.0" << endl;
        }
    }

    out << "VECTORS Velocity double" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            DAT ux_l, ux_r, uy_l, uy_u;
            ux_l = ux_r = uy_u = uy_l = 0.0;
            if(i > 0)
                ux_l = Vx[i-1+j*(nx+1)];
            if(i < nx)
                ux_r = Vx[i+1+j*(nx+1)];
            if(j > 0)
                uy_l = Vy[i+(j-1)*nx];
            if(j < ny)
                uy_u = Vy[i+(j+1)*nx];
            out << 0.5*(ux_l+ux_r) << " " <<
                   0.5*(uy_u+uy_l) << " 0.0" << endl;
        }
    }

    out << "SCALARS Rmx double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << rsd_m_x[i+j*nx] << endl;
        }
    }

    out << "SCALARS Rmy double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << rsd_m_y[i+j*nx] << endl;
        }
    }

    out.close();
}
