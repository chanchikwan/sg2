#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

void usage(int status)
{
  if(status)
    fprintf(stderr, "Try `ihd --help' for more information.\n");
  else
    fprintf(stdout, "Usage: ihd [OPTION...] [INPUT_FILE]\n\
2D spectral hydrodynamic code with CUDA\n\
\n\
      --help        display this help and exit\n\
  -n                kinematic viscosity\n\
  -m                Ekman coefficient\n\
  -f                forcing amplitude\n\
  -k                forcing wavenumber\n\
  -t                total time\n\
  -s                number of frames and grids\n\
  -o                prefix of the outputs\n\
\n\
Report bugs to <ckch@nordita.org>.\n");

  exit(status);
}
