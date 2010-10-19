#include <stdio.h>
#include <math.h>
#include "ihd.h"

R w0(R x, R y)
{
  return sin(x) * cos(y);
}

int main(int argc, char *argv[])
{
  Z n0 = (argc > 1) ? atoi(argv[1]) : 1024;
  Z n1 = (argc > 2) ? atoi(argv[2]) : 1024;
  Z n2 = (argc > 3) ? atoi(argv[3]) : 1024;

  R dt = (argc > 4) ? atof(argv[4]) : 1.0e-3;
  R nu = (argc > 5) ? atof(argv[5]) : 1.0e-3;

  Z i  = 0;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  printf("2D spectral hydrodynamic code with CUDA\n");
  setup(n1, n2);

  forward(W, init(w, w0));
  dump(i, inverse(w, W));

  while(i++ < n0) {
    float ms;
    printf("%4d: ", i);

    cudaEventRecord(t0, 0);
    /* TODO: step */
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
