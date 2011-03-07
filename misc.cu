#include <stdio.h>
#include "ihd.h"

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

int frame(const char *h)
{
  char c;
  while(c = *h++)
    if('0' <= c && c <= '9')
      return atoi(h-1); /* get the frame number */
  return 0;
}

const char *name(Z i)
{
  static char n[256];
  sprintf(n, "%s%04d.raw", Prefix, i);
  return n;
}
