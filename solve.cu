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
      R dt = getdt(nu, mu, fi);
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

    dump(name(i), W);
  }

  printf("======================= Done  Simulation =======================\n");

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);
  return 0;
}
