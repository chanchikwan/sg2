#include <stdio.h>
#include <math.h>
#include "ihd.h"

R w0(R x, R y)
{
  return sin(x) * cos(y);
}

int main(int argc, char *argv[])
{
  Z n0 = (argc > 1) ? atoi(argv[1]) : 1024;
  Z n1 = (argc > 2) ? atoi(argv[2]) : 1024;
  Z n2 = (argc > 3) ? atoi(argv[3]) : 1024;
  Z i  = 0;

  printf("2D spectral hydrodynamic code with CUDA\n");
  setup(n1, n2);

  forward(W, init(w, w0));
  dump(i, inverse(w, W));

  while(i++ < n0) {
    printf("%4d: \n", i);
    dump(i, inverse(w, W));
  }

  return 0;
}
