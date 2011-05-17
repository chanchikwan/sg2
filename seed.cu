#include <stdlib.h>
#include <stdio.h>
#include "sg2.h"

static Z rseed, byhand = 0;

Z setseed(const char *s)
{
  FILE *f = fopen(s, "r");
  if(!f) {
    char c;
    while((int)(c = *s++))
      if(c < '0' || '9' < c)
        return -1;
    byhand = 1;
    srand(rseed = atoi(s));
    return rseed;
  }
  fclose(f);
  return -1;
}

Z setseed(Z s)
{
  if(!byhand)
    srand(rseed = s);
  return rseed;
}

Z getseed(void)
{
  srand(rseed = rand());
  return rseed;
}
