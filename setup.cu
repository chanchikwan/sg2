#include <stdlib.h>
#include "ihd.h"

Z N1, N2, H2, F2;
R *w, *Host;
C *W;

static void done(void)
{
  rmplans();

  cudaFree(W);
  cudaFree(w);

  free(Host);
}

void setup(Z n1, Z n2)
{
  atexit(done);

  N1 = n1;
  N2 = n2;

  H2 = n2 / 2 + 1; /* number of non-redundant coefficients */
  F2 = H2 * 2;     /* necessary for in-place transform     */

  Host = (R *)malloc(sizeof(R) * N1 * N2);

  cudaMalloc(&w, sizeof(R) * N1 * N2);
  cudaMalloc(&W, sizeof(C) * N1 * H2);

  mkplans(n1, n2);
}
