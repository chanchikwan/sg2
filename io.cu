#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

R *load(R *f, const char *name)
{
  FILE *file;
  Z     size[4];

  file = fopen(name, "rb");
  fread(size, sizeof(Z), 4, file);

  if(size[0] == sizeof(R) && size[1] == N1 && size[2] == N2) {
    fread(Host, sizeof(R), N1 * N2, file);
    fclose(file);
    cudaMemcpy(f, Host, sizeof(R) * N1 * N2, cudaMemcpyHostToDevice);
    srand(Seed = size[3]);
    return f;
  } else {
    fclose(file);
    return NULL;
  }
}

R *dump(const char *name, R *f)
{
  FILE *file;
  Z     size[4] = {sizeof(R), N1, N2, Seed};

  cudaMemcpy(Host, f, sizeof(R) * N1 * N2, cudaMemcpyDeviceToHost);

  file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 4,       file);
  fwrite(Host, sizeof(R), N1 * N2, file);
  fclose(file);

  return f;
}
