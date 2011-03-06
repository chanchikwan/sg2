#include <stdio.h>
#include "ihd.h"

const char *name(Z i)
{
  static char n[256];
  sprintf(n, "%s%04d.raw", Prefix, i);
  return n;
}

int exist(const char *name)
{
  FILE *file = fopen(name, "r");
  if(file) {
    Z size[3];
    fread(size, sizeof(Z), 3, file);
    if(size[0] == sizeof(R) &&
       size[1] == N1        &&
       size[2] == N2          )
      return 1;
  }
  return 0;
}
