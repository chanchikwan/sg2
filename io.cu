#include <stdio.h>
#include "ihd.h"

R *load(R *f, const char *name)
{
  FILE *file;
  Z     size[3];

  file = fopen(name, "rb");
  fread(size, sizeof(Z), 3,       file); /* TODO: check data type and size */
  fread(Host, sizeof(R), N1 * N2, file);
  fclose(file);

  cudaMemcpy(f, Host, sizeof(R) * N1 * N2, cudaMemcpyHostToDevice);

  return f;
}

R *dump(const char *name, R *f)
{
  FILE *file;
  Z     size[3] = {sizeof(R), N1, N2};

  cudaMemcpy(Host, f, sizeof(R) * N1 * N2, cudaMemcpyDeviceToHost);

  file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 3,       file);
  fwrite(Host, sizeof(R), N1 * N2, file);
  fclose(file);

  return f;
}
