#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

void usage(int status)
{
  if(status)
    fprintf(stderr, "Try `ihd --help' for more information.\n");
  else
    fprintf(stdout, "Usage: ihd [OPTION...] [INPUT_FILE]\n\
Spectral Galerkin Incompressible Hydrodynamic in 2D (with CUDA)\n\
\n\
      --help        display this help and exit\n\
  -c                Courant number for the CFL condition\n\
  -d                device id\n\
  -f                forcing amplitude\n\
  -k                forcing wavenumber\n\
  -m                Ekman coefficient\n\
  -n                kinematic viscosity\n\
  -o                prefix of the outputs\n\
  -r                random number seed\n\
  -s                number of frames and grids\n\
  -t                total time\n\
\n\
Report bugs to <ckch@nordita.org>.\n");

  exit(status);
}
