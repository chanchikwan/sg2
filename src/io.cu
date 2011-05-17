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
#include "sg2.h"

C *load(C *F, const char *name)
{
  const Z k = (MIN(N1, N2) - 1) / 3;
  Z i, j, size[4];

  FILE *file = fopen(name, "rb");
  fread(size, sizeof(Z), 4, file);

  if(size[0] == -(Z)sizeof(C) &&
     size[1] <= 1 + k * 2     &&
     size[2] <= 1 + k         ) {
    const Z n1 = size[1];
    const Z h2 = size[2];
    const C zero = {0.0, 0.0};

    fread(Host, sizeof(C), n1 * h2, file);
    fclose(file);

    for(i = 1; i <= k; ++i) {
      for(j = H2-1; j >  k; --j) Host[(N1-i) * H2 + j] = zero;
      for(j = k   ; j >= 0; --j) Host[(N1-i) * H2 + j] = Host[(n1-i) * h2 + j];
    }

    for(i = (N1-k) * H2 - 1; i >= (k+1) * H2; --i) Host[i] = zero;

    for(i = k; i >= 0; --i) {
      for(j = H2-1; j >  k; --j) Host[i * H2 + j] = zero;
      for(j = k   ; j >= 0; --j) Host[i * H2 + j] = Host[i * h2 + j];
    }

    cudaMemcpy(F, Host, sizeof(C) * N1 * H2, cudaMemcpyHostToDevice);
    setseed(size[3]);
    return F;
  } else {
    fclose(file);
    return NULL;
  }
}

C *dump(const char *name, C *F)
{
  const Z k  = (MIN(N1, N2) - 1) / 3;
  const Z n1 = 1 + k * 2;
  const Z h2 = 1 + k;
  Z i, j, size[4] = {-(Z)sizeof(C), n1, h2, getseed()};

  FILE *file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 4, file);

  /* Write the Fourier space vorticity */
  cudaMemcpy(Host, F, sizeof(C) * N1 * H2, cudaMemcpyDeviceToHost);
  for(i = 0; i <= k; ++i)
    for(j = 0; j <= k; ++j)
      Host[i * h2 + j] = Host[i * H2 + j];
  for(i = k; i >= 1; --i)
    for(j = 0; j <= k; ++j)
      Host[(n1 - i) * h2 + j] = Host[(N1 - i) * H2 + j];
  fwrite(Host, sizeof(C), n1 * h2, file);

  /* Compute the non-linear term */
  scale(w, 0.0);
  jacobi1(X, Y, W); add_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));
  jacobi2(X, Y, W); sub_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));
  scale(forward(X, w), 1.0 / (N1 * N2));

  /* Write the Fourier space non-linear term */
  cudaMemcpy(Host, X, sizeof(C) * N1 * H2, cudaMemcpyDeviceToHost);
  for(i = 0; i <= k; ++i)
    for(j = 0; j <= k; ++j)
      Host[i * h2 + j] = Host[i * H2 + j];
  for(i = k; i >= 1; --i)
    for(j = 0; j <= k; ++j)
      Host[(n1 - i) * h2 + j] = Host[(N1 - i) * H2 + j];
  fwrite(Host, sizeof(C), n1 * h2, file);

  fclose(file);

  return F;
}
