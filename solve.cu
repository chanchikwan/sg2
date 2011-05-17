#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ihd.h"

int solve(R nu, R mu, R fi, R ki, R dT, Z i, Z n)
{
  const char rotor[] = "-/|\\";
  const R flop = N1 * N2 * SUBS * (fi * ki > 0.0 ? 0 : 8) +
    N1 * N2 * SUBS * (21.5 + 12.5 * (log2((double)N1) + log2((double)N2))) +
    N1 * N2 * (4.0 + 5.0 * (log2((double)N1) + log2((double)N2)));
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
    srand(Seed);

    cudaEventRecord(t0, 0);
    while(time < next) {
      R dt = getdt(nu, mu, fi);
      if(dt == 0.0) {
        fflush(stdout);
        fprintf(stderr, "\ndiverged, QUIT\n");
        exit(-1);
      }
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
           m > 1 ? 's' : ' ', ms, 1e-6 * flop / ms);

    dump(name(i), W);
  }

  printf("======================= Done  Simulation =======================\n");

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);
  return 0;
}
