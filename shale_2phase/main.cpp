#include "header.h"

int main(int argc, char *argv[])
{
    printf("Ready to create problem\n");
    fflush(stdout);
    Problem P;
    P.SolveOnGPU();
    return 0;
}
