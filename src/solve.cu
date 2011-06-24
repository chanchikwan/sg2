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

int solve(R nu, R mu, R fi, R ki, R dT, Z i, Z n)
{
  const char rotor[] = "-/|\\";

  R time = dT * i;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  printf("======================= Start Simulation =======================\n");

  while(++i <= n) {
    const R next = dT * i;
    Z m = 0;
    float ms;
    printf("%4d: %5.2f -> %5.2f:                  ", i, time, next);

    cudaEventRecord(t0, 0);
    while(time < next) {
      R dt = getdt(nu, mu, fi, time);
      if(dt == 0.0) error(" diverged, QUIT\n");
      printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%5d ", ++m);
      if(time + dt < next)
        time += dt;
      else {
        dt = next - time;
        time = next;
      }
      printf("%c dt ~ %5.0e", rotor[m%4], dt);
      fflush(stdout);
      step(nu, mu, fi, ki, dt);
    }
    cudaEventRecord(t1, 0);

    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1); ms /= m;
    printf("\b\b\b\b\b\b\b\b\b\b\b\bstep%c %7.3f ms/cycle ~ %.3f GFLOPS\n",
           m > 1 ? 's' : ' ', ms, 1e-6 * flop() / ms);

    dump(name(i, "raw"), W);
  }

  printf("======================= Done  Simulation =======================\n");

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);
  return 0;
}
