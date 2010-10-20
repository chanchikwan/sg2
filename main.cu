#include <stdio.h>
#include <math.h>
#include "ihd.h"

R w0(R x, R y)
{
  return sin(x) * cos(y);
}

int main(int argc, char *argv[])
{
  Z ns = 32; /* number of substeps */

  Z n0 = (argc > 1) ? atoi(argv[1]) : 2048 / ns;
  Z n1 = (argc > 2) ? atoi(argv[2]) : 1024;
  Z n2 = (argc > 3) ? atoi(argv[3]) : 1024;

  R dt = (argc > 4) ? atof(argv[4]) : TWO_PI / (n0 * ns);
  R nu = (argc > 5) ? atof(argv[5]) : 0.0;

  Z i  = 0, j;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  printf("2D spectral hydrodynamic code with CUDA\n");
  setup(n1, n2);

  scale(forward(W, init(w, w0)), 1.0f / (n1 * n2));
  dump(i, inverse(w, W));

  while(i++ < n0) {
    float ms;
    printf("%4d: ", i);

    cudaEventRecord(t0, 0);
    for(j = 0; j < ns; ++j) step(nu, dt);
    cudaEventRecord(t1, 0);

    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("%.3f ms\n", (double)ms);

    dump(i, inverse(w, W));
  }

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);

  return 0;
}
