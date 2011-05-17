#include <stdio.h>
#include <string.h>
#include "ihd.h"

#define SIZE 256

static char prefix[SIZE] = "";

void setprefix(const char *p)
{
  strncat(prefix, p, SIZE);
  prefix[SIZE - 1] = '\0';
}

int exist(const char *name)
{
  const Z k  = (MIN(N1, N2) - 1) / 3;

  FILE *file = fopen(name, "r");
  if(file) {
    Z size[3];
    fread(size, sizeof(Z), 3, file);

  if(size[0] == -(Z)sizeof(C) &&
     size[1] <= 1 + k * 2     &&
     size[2] <= 1 + k         )
      return 1;
  }
  return 0;
}

int seed(const char *s)
{
  FILE *f = fopen(s, "r");
  if(!f) {
    char c;
    while((int)(c = *s++))
      if(c < '0' || '9' < c)
        return 0;
    return 1;
  }
  fclose(f);
  return 0;
}

int frame(const char *h)
{
  char c;
  while((int)(c = *h++))
    if('0' <= c && c <= '9')
      return atoi(h-1); /* get the frame number */
  return 0;
}

const char *name(Z i)
{
  static char n[SIZE];
  snprintf(n, SIZE, "%s%04d.raw", prefix, i);
  return n;
}
