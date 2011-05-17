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

#include "sg2.h"

#define KERN(NAME, OP)                                  \
  static __global__ void NAME(C *X, C *Y, const C *W,   \
                              const Z N1, const Z H2)   \
  {                                                     \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;  \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;  \
    const Z h = i * H2 + j;                             \
                                                        \
    if(i < N1 && j < H2) {                              \
      const C Wh = W[h];                                \
      R kx = i < N1 / 2 ? i : i - N1;                   \
      R ky = j;                                         \
      if(h) {                                           \
        const R ikk = K(1.0) / (kx * kx + ky * ky);     \
        OP;                                             \
      }                                                 \
                                                        \
      X[h].r = - kx * Wh.i;                             \
      X[h].i =   kx * Wh.r;                             \
      Y[h].r = - ky * Wh.i;                             \
      Y[h].i =   ky * Wh.r;                             \
    }                                                   \
  }

/* Compute dF/dx and dF/dy from W */
KERN(_Fx_Fy, kx *= ikk; ky *= ikk)

void getu(C *Fx, C *Fy, const C *W)
{
  _Fx_Fy<<<Hsz, Bsz>>>(Fx, Fy, W, N1, H2);
}

/* Compute dF/dx and dW/dy from W for the 1st term in the Jacobian */
KERN(_Fx_Wy, kx *= ikk)

void jacobi1(C *Fx, C *Wy, const C *W)
{
  _Fx_Wy<<<Hsz, Bsz>>>(Fx, Wy, W, N1, H2);
}

/* Compute dW/dx and dF/dy from W for the 2nd term in the Jacobian */
KERN(_Wx_Fy, ky *= ikk)

void jacobi2(C *Wx, C *Fy, const C *W)
{
  _Wx_Fy<<<Hsz, Bsz>>>(Wx, Fy, W, N1, H2);
}
