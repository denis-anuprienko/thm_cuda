#include "header.h"
#include <filesystem>

using namespace std;

void Problem::Init()
{
    //cout << "Save files every " << save_intensity << " step" << endl;
    respath = std::experimental::filesystem::current_path().string() + "\\res";
    printf("c_f = %e, c_phi = %e\n", c_f, c_phi);
    fflush(stdout);
    flog.open(respath + "\\log.txt");
    flog << "nx = " << nx << ", ny = " << ny << ", nt = " << nt << ", dt = " << dt << ", niter = " << niter << endl;
    DAT D      = 1e-18/min(mul,mug);// * max(rhol,rhog) * max(Krl,Krg);
    DAT dtau_t = 1./(4.1*D/min(dx*dx,dy*dy) + 1./dt);
    DAT dtau   = min(dx*dx,dy*dy) / D / 4.1 * 0.16*rhol;
    printf("dt = %e, dtau = %e, dtau_t = %e\n", dt, dtau, dtau_t);
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

    for(int j = 0; j <= ny; j++){
        for(int i = 0; i <= nx; i++){
            out << i*dx << " " << j*dy << " 0.0" << endl;
        }
    }

    out << "CELL_DATA " << nx * ny << endl;

//    out << "SCALARS Fluid_Pressure double" << endl;
//    out << "LOOKUP_TABLE default" << endl;
//    for(int j = 0; j < ny; j++){
//        for(int i = 0; i < nx; i++){
//            out << Sl[i+j*nx]*Pl[i+j*nx] + (1.0-Sl[i+j*nx])*Pg[i+j*nx] << endl;
//        }
//    }

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

    out << "SCALARS Gas_Pressure double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Pg[i+j*nx] << endl;
        }
    }

//    out << "SCALARS Gas_Saturation double" << endl;
//    out << "LOOKUP_TABLE default" << endl;
//    for(int j = 0; j < ny; j++){
//        for(int i = 0; i < nx; i++){
//            out << 1.-Sl[i+j*nx] << endl;
//        }
//    }

    out << "SCALARS Capillary_Pressure double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << Pg[i+j*nx]-Pl[i+j*nx] << endl;
        }
    }

    out << "SCALARS Porosity double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << phi[i+j*nx] << endl;
        }
    }

//    out << "SCALARS Cell_Number double" << endl;
//    out << "LOOKUP_TABLE default" << endl;
//    for(int j = 0; j < ny; j++){
//        for(int i = 0; i < nx; i++){
//            out << i+j*nx << endl;
//        }
//    }

//    out << "VECTORS Intrinsic_Permeability double" << endl;
//    for(int j = 0; j < ny; j++){
//        for(int i = 0; i < nx; i++){
//            out << 0.5*(Kx[i+j*(nx+1)]+Kx[i+1+j*(nx+1)]) << " " <<
//                   0.5*(Ky[i+(j+1)*nx]+Ky[i+j*nx]) << " 0.0" << endl;
//        }
//    }

//    out << "VECTORS Liquid_Flux double" << endl;
//    for(int j = 0; j < ny; j++){
//        for(int i = 0; i < nx; i++){
//            out << 0.5*(qlx[i+j*(nx+1)]+qlx[i+1+j*(nx+1)]) << " " <<
//                   0.5*(qly[i+(j+1)*nx]+qly[i+j*nx]) << " 0.0" << endl;
//        }
//    }

    out << "SCALARS Rl double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << rsd_l[i+j*nx] << endl;
        }
    }
    out << "SCALARS Rg double" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            out << rsd_g[i+j*nx] << endl;
        }
    }

    out.close();
}
