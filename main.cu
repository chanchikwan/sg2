#include <stdio.h>
#include "ihd.h"

int main(int argc, char *argv[])
{
  Z n1 = (argc > 1) ? atoi(argv[1]) : 1024;
  Z n2 = (argc > 2) ? atoi(argv[2]) : 1024;

  printf("2D spectral hydrodynamic code with CUDA\n");

  setup(n1, n2);

  return 0;
}
