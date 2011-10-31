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
#include <math.h>
#include "sg2.h"

/* We turn on Kolmogorov forcing when fi * ki < 0.  Therefore, there
   is an extra minus sign in line 19 although we are using

     f_K = fi ki cos(ki x)

   as the forcing. */
static __global__ void _force(R *f, const R sl, const R fi, const R ki,
                                    const Z N1, const Z N2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * N2 + j;

  if(i < N1 && j < N2) {
    const R dx = K(1.0) / N1;
    f[h] = sl * f[h] - fi * ki * cos(TWO_PI * ki * i * dx);
  }
}

R *force(R *f, R sl, R fi, R ki)
{
  _force<<<Gsz, Bsz>>>(f, sl, fi, ki, N1, N2);
  return f;
}

/* When ky != 0, the random forcing is straightforward. */
static __global__ void _force1(C *F, const R fx, const R fy,
                                     const Z kx, const Z ky,
                                     const Z N1, const Z H2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * H2 + j;

  if(i < N1 && j < H2) {
    const Z k = i < N1 / 2 ? i : i - N1;
    if(k == kx && (j == ky || j == -ky)) {
      F[h].r += fx;
      F[h].i += fy;
    }
  }
}

/* After transforming along the y-direction, the ky == 0 column is
   real.  It is necessary to implement the Hermit symmetric by hand.
   The force is added to both the k == kx and k ==-kx modes. */
static __global__ void _force2(C *F, const R fx, const R fy,
                                     const Z kx, const Z ky,
                                     const Z N1, const Z H2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * H2 + j;

  if(i < N1 && j < H2) {
    const Z k = i < N1 / 2 ? i : i - N1;
    if(k == kx && j == ky) {
      F[h].r += fx;
      F[h].i += fy;
    }
    if(k ==-kx && j == ky) {
      F[h].r += fx;
      F[h].i -= fy;
    }
  }
}

C *force(C *F, R dt, R fi, R ki)
{
  const R fs = fi * ki * sqrt(dt); /* no factor of 2 because of FFT */
  const R dp = TWO_PI / (RAND_MAX + 1.0);
  const R pm = dp * rand();
  const R pk = dp * rand();

  const R fx = fs * cos(pm);
  const R fy = fs * sin(pm);
  const Z kx = (Z)floor(ki * cos(pk) + 0.5);
  const Z ky = (Z)floor(ki * sin(pk) + 0.5);

  if(ky) _force1<<<Hsz, Bsz>>>(F, fx, fy, kx, ky, N1, H2);
  else   _force2<<<Hsz, Bsz>>>(F, fx, fy, kx, ky, N1, H2);

  return F;
}
