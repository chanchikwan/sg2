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

#define ZERO(T, OP)                                             \
  static __global__ void _zero(T *f, const Z N1, const Z H2)    \
  {                                                             \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;          \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;          \
    const Z h = i * H2 + j;                                     \
                                                                \
    if(i < N1 && j < H2) {                                      \
      OP;                                                       \
    }                                                           \
  }

#define SCAL(T, OP)                                                     \
  static __global__ void _scal(T *f, const R s, const Z N1, const Z H2) \
  {                                                                     \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;                  \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;                  \
    const Z h = i * H2 + j;                                             \
                                                                        \
    if(i < N1 && j < H2) {                                              \
      OP;                                                               \
    }                                                                   \
  }

ZERO(C, f[h].i = f[h].r = K(0.0))
SCAL(C, f[h].r *= s; f[h].i *=s)

C *scale(C *F, R s)
{
  if(s == 0.0) _zero<<<Hsz, Bsz>>>(F,    N1, H2);
  else         _scal<<<Hsz, Bsz>>>(F, s, N1, H2);
  return F;
}

ZERO(R, f[h] = K(0.0))
SCAL(R, f[h] *= s)

R *scale(R *f, R s)
{
  if(s == 0.0) _zero<<<Gsz, Bsz>>>(f,    N1, N2);
  else         _scal<<<Gsz, Bsz>>>(f, s, N1, N2);
  return f;
}
