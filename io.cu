#include <stdio.h>
#include "ihd.h"

R *load(R *f, Z i)
{
  char  name[256];
  FILE *file;
  Z     size[3];

  sprintf(name, "%04d.raw", i);
  file = fopen(name, "rb");
  fread(size, sizeof(Z), 3,       file); /* TODO: check data type and size */
  fread(Host, sizeof(R), N1 * N2, file);
  fclose(file);

  cudaMemcpy(f, Host, sizeof(R) * N1 * N2, cudaMemcpyHostToDevice);

  return f;
}

Z dump(Z i, R *f)
{
  char  name[256];
  FILE *file;
  Z     size[3] = {sizeof(R), N1, N2};

  cudaMemcpy(Host, f, sizeof(R) * N1 * N2, cudaMemcpyDeviceToHost);

  sprintf(name, "%04d.raw", i);
  file = fopen(name, "wb");
  fwrite(size, sizeof(Z), 3,       file);
  fwrite(Host, sizeof(R), N1 * N2, file);
  fclose(file);

  return i;
}
