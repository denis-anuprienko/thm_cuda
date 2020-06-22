#include "header.h"
#include <filesystem>

using namespace std;

void Problem::Init()
{
    cout << "Save files every " << save_intensity << " step" << endl;
    respath = std::experimental::filesystem::current_path().string() + "\\res";
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

    for(int i = 0; i <= nx; i++){
        for(int j = 0; j <= ny; j++){
            out << j*dy << " " << i*dx << " 0.0" << endl;
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

    out << "VECTORS FluidFlux double" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << 0.5*(qx[i+j*(nx+1)]+qx[i+1+j*(nx+1)]) << " " <<
                   0.5*(qy[i+(j+1)*nx]+qy[i+j*nx]) << " 0.0" << endl;
        }
    }

    out.close();
}
