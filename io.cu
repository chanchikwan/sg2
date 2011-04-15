#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

C *load(C *F, const char *name)
{
  FILE *file;
  Z     size[4];

  file = fopen(name, "rb");
  fread(size, sizeof(Z), 4, file);

  if(size[0] == -(Z)sizeof(C) &&
     size[1] == N1            &&
     size[2] == H2            ) {
    fread(Host, sizeof(C), N1 * H2, file);
    fclose(file);
    cudaMemcpy(F, Host, sizeof(C) * N1 * H2, cudaMemcpyHostToDevice);
    srand(Seed = size[3]);
    return F;
  } else {
    fclose(file);
    return NULL;
  }
}

C *dump(const char *name, C *F)
{
  FILE *file;
  Z     size[4] = {-(Z)sizeof(C), N1, H2, Seed};

  cudaMemcpy(Host, F, sizeof(C) * N1 * H2, cudaMemcpyDeviceToHost);

  file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 4,       file);
  fwrite(Host, sizeof(C), N1 * H2, file);
  fclose(file);

  return F;
}
