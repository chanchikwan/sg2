#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

C *load(C *F, const char *name)
{
  const Z k  = (MIN(N1, N2) - 1) / 3;

  FILE *file;
  Z i,j,size[4];

  file = fopen(name, "rb");
  fread(size, sizeof(Z), 4, file);

  if(size[0] == -(Z)sizeof(C) &&
     size[1] <= 1 + k * 2     &&
     size[2] <= 1 + k         ) {
    const Z n1 = size[1];
    const Z h2 = size[2];
    const C zero = {0.0, 0.0};

    fread(Host, sizeof(C), n1 * h2, file);
    fclose(file);

    for(i = 1; i <= k; ++i) {
      for(j = H2-1; j >  k; --j) Host[(N1-i) * H2 + j] = zero;
      for(j = k   ; j >= 0; --j) Host[(N1-i) * H2 + j] = Host[(n1-i) * h2 + j];
    }

    for(i = (N1-k) * H2 - 1; i >= (k+1) * H2; --i) Host[i] = zero;

    for(i = k; i >= 0; --i) {
      for(j = H2-1; j >  k; --j) Host[i * H2 + j] = zero;
      for(j = k   ; j >= 0; --j) Host[i * H2 + j] = Host[i * h2 + j];
    }

    cudaMemcpy(F, Host, sizeof(C) * N1 * H2, cudaMemcpyHostToDevice);
    Seed = size[3];
    return F;
  } else {
    fclose(file);
    return NULL;
  }
}

C *dump(const char *name, C *F)
{
  const Z k  = (MIN(N1, N2) - 1) / 3;
  const Z n1 = 1 + k * 2;
  const Z h2 = 1 + k;

  FILE *file;
  Z i,j,size[4] = {-(Z)sizeof(C), n1, h2, Seed};

  cudaMemcpy(Host, F, sizeof(C) * N1 * H2, cudaMemcpyDeviceToHost);

  for(i = 0; i <= k; ++i)
    for(j = 0; j <= k; ++j)
      Host[i * h2 + j] = Host[i * H2 + j];

  for(i = k; i >= 1; --i)
    for(j = 0; j <= k; ++j)
      Host[(n1 - i) * h2 + j] = Host[(N1 - i) * H2 + j];

  file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 4,       file);
  fwrite(Host, sizeof(C), n1 * h2, file);
  fclose(file);

  return F;
}
