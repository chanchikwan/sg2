#include "ihd.h"

R *init(R *f, R (*func)(R, R))
{
  const R d1 = TWO_PI / N1;
  const R d2 = TWO_PI / N2;

  Z i, j;
  for(i = 0; i < N1; ++i)
    for(j = 0; j < N2; ++j)
      Host[i * N2 + j] = func(d1 * i, d2 * j);

  cudaMemcpy(f, Host, sizeof(R) * N1 * N2, cudaMemcpyHostToDevice);

  return f;
}
