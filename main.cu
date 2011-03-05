#include <stdio.h>
#include <math.h>
#include "ihd.h"

R noise(R x, R y)
{
  return 0.5 - (R)rand() / RAND_MAX;
}

R decay(R x, R y)
{
  return 1024.0 * noise(x, y); /* so u = curl(w) ~ 1 */
}

R KH(R x, R y)
{
  return noise(x, y) + (fabs(x - 0.0 ) < 1.0e-6 ? -512.0 : 0.0)
                     + (fabs(x - M_PI) < 1.0e-6 ?  512.0 : 0.0);
}

int main(int argc, char *argv[])
{
  const char rotor[] = "-/|\\";

  R nu = (argc > 1) ? atof(argv[1]) : 1.0e-4;
  R mu = (argc > 2) ? atof(argv[2]) : 1.0e-2;
  R tt = (argc > 3) ? atof(argv[3]) : 1.0e+2;

  Z n0 = (argc > 4) ? atoi(argv[4]) : 1024;
  Z n1 = (argc > 5) ? atoi(argv[5]) : 1024;
  Z n2 = (argc > 6) ? atoi(argv[6]) : 1024;

  R fo = 5 * n1 * n2 * (21.5 + 12.5 * (log2((double)n1) + log2((double)n2)));
  Z i  = 0;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  printf("2D spectral hydrodynamic code with CUDA\n");
  setup(n1, n2);

  scale(forward(W, init(w, KH)), 1.0 / (n1 * n2));
  dump(i, inverse(w, W));

  while(i++ < n0) {
    float ms;
    Z ns = (Z)ceil(tt / n0 / 0.9 / getdt(10.0, nu, mu)), j;
    R dt =         tt / n0 / ns;
    printf("%4d: %5.2f -> %5.2f, dt ~ %.0e:       ",
           i, dt * ns * (i-1), dt * ns * i, dt);

    cudaEventRecord(t0, 0);
    for(j = 0; j < ns; ++j) {
      printf("\b\b\b\b\b\b%c %4d", rotor[j%4], j+1);
      fflush(stdout);
      step(nu, mu, dt);
    }
    cudaEventRecord(t1, 0);

    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1); ms /= ns;
    printf("\b\b\b\b\b\b%.3f ms/cycle ~ %.3f GFLOPS\n",
           ms, 1e-6 * fo / ms);

    dump(i, inverse(w, W));
  }

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);

  return 0;
}
