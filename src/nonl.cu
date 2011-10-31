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

#include "sg2.h"

#define KERN(NAME, OP)                                                  \
  static __global__ void NAME(R *f, const R *x, const R *y, const R z,  \
                                    const Z N1, const Z N2, const Z F2) \
  {                                                                     \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;                  \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;                  \
    const Z h = i * N2 + j;                                             \
    const Z H = i * F2 + j;                                             \
                                                                        \
    if(i < N1 && j < N2) f[h] OP x[H] * y[H];                           \
  }

/* Adding the 1st term in the Jacobian */
KERN(_add_pro, += z * x[H] +)

R *add_pro(R *f, const R *x, const R *y, R beta)
{
  _add_pro<<<Gsz, Bsz>>>(f, x, y, beta, N1, N2, F2);
  return f;
}

/* Adding the 2nd term in the Jacobian */
KERN(_sub_pro,  = z * f[h] -)

R *sub_pro(R *f, const R *x, const R *y, R scale)
{
  _sub_pro<<<Gsz, Bsz>>>(f, x, y, scale, N1, N2, F2);
  return f;
}
