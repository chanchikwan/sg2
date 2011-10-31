/* Copyright (C) 2010-2011 Chi-kwan Chan
   Copyright (C) 2010-2011 NORDITA

   This file is part of sg2.

   Sg2 is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Sg2 is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with sg2.  If not, see <http://www.gnu.org/licenses/>. */

#include <stdlib.h>
#include <stdio.h>
#include "sg2.h"

void usage(int status)
{
  if(status)
    fprintf(stderr, "Try `ihd --help' for more information.\n");
  else
    fprintf(stdout, "Usage: ihd [OPTION...] [SEED/INPUT_FILE]\n\
Spectral Galerkin Incompressible Hydrodynamic in 2D (with CUDA)\n\
\n\
      --help        display this help and exit\n\
  -b                quasi-geostrophy beta parameter\n\
  -d                device id\n\
  -f                forcing [amplitude and] wavenumber\n\
  -m                Ekman coefficient\n\
  -n                kinematic viscosity\n\
  -o                prefix of the outputs\n\
  -rk3, -rk4        pick different time integrators\n\
  -s                number of frames and grids\n\
  -t                [Courant number and] total time [and fixed step size]\n\
\n\
Report bugs to <ckch@nordita.org>.\n");

  exit(status);
}
