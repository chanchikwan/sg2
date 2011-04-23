#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ihd.h"

static R typical_v = 1.0;

int solve(R nu, R mu, R fi, R ki, R tt, Z i, Z n)
{
  const char rotor[] = "-/|\\";
  const R flop = 5 * N1 * N2 * (21.5 + (fi * ki > 0.0 ? 0 : 8) +
                                12.5 * (log2((double)N1) + log2((double)N2)));
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  printf("======================= Start Simulation =======================\n");

  for(++i; i <= n; ++i) {
    float ms;
    Z ns = (Z)ceil(tt / n / getdt(typical_v, nu, mu)), j;
    R dt =         tt / n / ns;
    printf("%4d: %5.2f -> %5.2f, dt ~ %.0e:       ",
           i, dt * ns * (i-1), dt * ns * i, dt);
    srand(Seed);

    cudaEventRecord(t0, 0);
    for(j = 0; j < ns; ++j) {
      printf("\b\b\b\b\b\b%c %4d", rotor[j%4], j+1);
      fflush(stdout);
      step(nu, mu, fi, ki, dt);
    }
    cudaEventRecord(t1, 0);

    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1); ms /= ns;
    printf("\b\b\b\b\b\b%.3f ms/cycle ~ %.3f GFLOPS\n",
           ms, 1e-6 * flop / ms);

    cudaMemcpy(Host, W, sizeof(R), cudaMemcpyDeviceToHost);
    if(Host[0].r == Host[0].r) /* spectrum is finite */
      dump(name(i), W);
    else if(exist(name(--i)) && load(W, name(i))) /* spectrum contains NAN */
      typical_v *= sqrt(2.0);
    else {
      fflush(stdout);
      fprintf(stderr, "diverged, fail to resume from \"%s\", QUIT\n", name(i));
      exit(-1);
    }
  }

  printf("======================= Done  Simulation =======================\n");

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);
  return 0;
}
