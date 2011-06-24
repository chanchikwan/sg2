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
   along with sg2. If not, see <http://www.gnu.org/licenses/>. */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "sg2.h"

/* If cfl == 0.0, fix is the fixed step */
static R cfl = CFL;
static R fix = DT_MIN;

static FILE *file = NULL;
static void done(void)
{
  if(file) fclose(file);
}

void setdt(R c, R f)
{
  cfl = c;
  fix = f;
}

R getdt(R nu, R mu, R fi, R t, Z i)
{
  static Z id = -1;
  R m, s;

  if(id == -1) atexit(done);

  if(id !=  i) { /* open new log file */
    done();
    file = fopen(name(i, "txt"), "w");
    id = i;
  }

  getu(X, Y, W);
  reduce(&m, &s, inverse((R *)X, X), inverse((R *)Y, Y));
  cudaMemcpy(Host,   W+1,  sizeof(C), cudaMemcpyDeviceToHost);
  cudaMemcpy(Host+1, W+H2, sizeof(C), cudaMemcpyDeviceToHost);
  if(Host->r != Host->r) return 0.0; /* spectrum contains NAN */

  fprintf(file, "%g %g %g %g %g %g\n", t, 0.5 * s / (N1 * N2),
          Host[0].r, Host[0].i, Host[1].r, Host[1].i);

  if(cfl == 0.0) return fix;
  else {
    const R n   = MIN(N1, N2);
    const R adv = 10.0 / (sqrt(m) * n);
    const R dff = 5.95 / (nu * n * n / 9.0 + mu);
    const R frc = pow(10.0 / fabs(fi * n), 2.0 / 3.0);
    const R dt  = cfl * MIN(adv, MIN(dff, frc));
    return dt < fix ? 0.0 : dt;
  }
}
