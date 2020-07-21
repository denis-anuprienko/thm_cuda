#include "header.h"
#include <filesystem>

using namespace std;

void Problem::Init()
{
    //cout << "Save files every " << save_intensity << " step" << endl;
    respath = std::experimental::filesystem::current_path().string() + "\\res";
    printf("Init finished\n");
    printf("c_f = %e, c_phi = %e\n", c_f, c_phi);
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

    for(int j = 0; j <= ny; j++){
        for(int i = 0; i <= nx; i++){
            out << i*dx << " " << j*dy << " 0.0" << endl;
        }
    }

    out << "CELL_DATA " << nx * ny << endl;

    out << "SCALARS Liquid_Pressure double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Pl[i+j*nx] << endl;
        }
    }

    out << "SCALARS Liquid_Saturation double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Sl[i+j*nx] << endl;
        }
    }

    out << "SCALARS Porosity double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << phi[i+j*nx] << endl;
        }
    }

    out << "SCALARS Cell_Number double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << i+j*nx << endl;
        }
    }

    out << "VECTORS Intrinsic_Permeability double" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << 0.5*(Kx[i+j*(nx+1)]+Kx[i+1+j*(nx+1)])/K0 << " " <<
                   0.5*(Ky[i+(j+1)*nx]+Ky[i+j*nx])/K0 << " 0.0" << endl;
        }
    }

    out.close();
}
