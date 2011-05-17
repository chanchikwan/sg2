#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

void usage(int status)
{
  if(status)
    fprintf(stderr, "Try `ihd --help' for more information.\n");
  else
    fprintf(stdout, "Usage: ihd [OPTION...] [SEED/INPUT_FILE]\n\
Spectral Galerkin Incompressible Hydrodynamic in 2D (with CUDA)\n\
\n\
      --help        display this help and exit\n\
  -d                device id\n\
  -f                forcing [amplitude and] wavenumber\n\
  -m                Ekman coefficient\n\
  -n                kinematic viscosity\n\
  -o                prefix of the outputs\n\
  -s                number of frames and grids\n\
  -t                [Courant number and] total time [and fixed step size]\n\
\n\
Report bugs to <ckch@nordita.org>.\n");

  exit(status);
}
