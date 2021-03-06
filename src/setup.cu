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
#include "sg2.h"

#define TIDE 32

Z N1, N2, H2, F2;
uint3 Bsz, Gsz, Hsz;

R *w, Flop;
C *W, *X, *Y, *Host;

static void done(void)
{
  rmplans();

  cudaFree(Y);
  cudaFree(X);
  cudaFree(W);
  cudaFree(w);

  free(Host);
}

void setup(Z n1, Z n2)
{
  cudaDeviceProp dev;
  Z m;

  atexit(done);

  cudaGetDeviceProperties(&dev, 0);
  m = dev.maxThreadsPerBlock;

  N1 = n1;
  N2 = n2;
  H2 = n2 / 2 + 1; /* number of non-redundant coefficients */
  F2 = H2 * 2;     /* necessary for in-place transform     */

  Bsz = make_uint3(TIDE, m / TIDE, 1);
  Gsz = make_uint3((N2 - 1) / Bsz.x + 1, (N1 - 1) / Bsz.y + 1, 1);
  Hsz = make_uint3((H2 - 1) / Bsz.x + 1, (N1 - 1) / Bsz.y + 1, 1);

  Host = (C *)malloc(sizeof(C) * N1 * H2);

  cudaMalloc(&w, sizeof(R) * N1 * F2); scale((C *)w, 0.0);
  cudaMalloc(&W, sizeof(C) * N1 * H2); scale((C *)W, 0.0);
  cudaMalloc(&X, sizeof(C) * N1 * H2); scale((C *)X, 0.0);
  cudaMalloc(&Y, sizeof(C) * N1 * H2); scale((C *)Y, 0.0);

  mkplans(n1, n2);
}
