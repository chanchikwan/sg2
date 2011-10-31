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
#include <string.h>
#include <math.h>
#include "sg2.h"

static R zeros(R x, R y)
{
  return 0.0;
}

static R noise(R x, R y)
{
  return 0.5 - (R)rand() / (RAND_MAX + 1.0);
}

static R decay(R x, R y)
{
  return 1024.0 * noise(x, y); /* so u = curl(w) ~ 1 */
}

static R KH(R x, R y)
{
  return noise(x, y) + (fabs(x - 0.0 ) < 1.0e-6 ? -512.0 : 0.0)
                     + (fabs(x - M_PI) < 1.0e-6 ?  512.0 : 0.0);
}

C *init(C *F, const char *name)
{
  const R d1 = TWO_PI / N1;
  const R d2 = TWO_PI / N2;

  R *h = (R *)Host;
  R *f = (R *)F;

  R (*func)(R, R);
  Z i, j;

       if(!strcmp(name, "zeros")) func = zeros;
  else if(!strcmp(name, "noise")) func = noise;
  else if(!strcmp(name, "decay")) func = decay;
  else if(!strcmp(name, "KH"   )) func = KH;
  else return NULL;

  for(i = 0; i < N1; ++i) {
    for(j = 0; j < N2; ++j) h[i * F2 + j] = func(d1 * i, d2 * j);
    for(     ; j < F2; ++j) h[i * F2 + j] = 0.0;
  }

  cudaMemcpy(f, h, sizeof(R) * N1 * F2, cudaMemcpyHostToDevice);
  return scale(forward(F, f), 1.0 / (N1 * N2));
}
