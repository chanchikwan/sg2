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

#define TIDE 512

static __global__ void _reduce(C *Out, const R *x, const R *y,
                                       const Z N1, const Z N2, const Z F2)
{
  __shared__ R max[TIDE], sum[TIDE];

  const Z t = threadIdx.x;
  const Z j = blockDim.x * blockIdx.x + t;
  Z i;

  max[t] = 0.0;
  sum[t] = 0.0;

  if(j < N2) for(i = 0; i < N1; ++i) {
    const Z h  = i * F2 + j;
    const R ux = x[h];
    const R uy = y[h];
    const R uu = ux * ux + uy * uy;
    if(max[t] < uu) max[t] = uu;
    sum[t] += uu;
  }
  __syncthreads();

  for(i = blockDim.x / 2; i > 0; i /= 2) {
    if(t < i) {
      if(max[t] < max[t + i]) max[t] = max[t + i];
      sum[t] += sum[t + i];
    }
    __syncthreads();
  }

  if(t == 0) {
    Out[blockIdx.x].r = max[0];
    Out[blockIdx.x].i = sum[0];
  }
}

void reduce(R *m, R *s, const R *ux, const R *uy)
{
  const Z bsz = TIDE;
  const Z gsz = (N2 - 1) / bsz + 1;

  R max = 0.0, sum = 0.0;
  Z i;

  _reduce<<<gsz, bsz>>>((C *)w, ux, uy, N1, N2, F2);
  cudaMemcpy(Host, w, sizeof(C) * gsz, cudaMemcpyDeviceToHost);

  for(i = 0; i < gsz; ++i) {
    if(max < Host[i].r) max = Host[i].r;
    sum += Host[i].i;
  }

  if(m) *m = max;
  if(s) *s = sum;
}
