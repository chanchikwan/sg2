#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ihd.h"

static R zeros(R x, R y)
{
  return 0.0;
}

static R noise(R x, R y)
{
  return 0.5 - (R)(Seed = rand()) / (RAND_MAX + 1.0);
}

static R decay(R x, R y)
{
  return 1024.0 * noise(x, y); /* so u = curl(w) ~ 1 */
}

static R KH(R x, R y)
{
  return noise(x, y) + (fabs(x - 0.0 ) < 1.0e-6 ? -512.0 : 0.0)
                     + (fabs(x - M_PI) < 1.0e-6 ?  512.0 : 0.0);
}

C *init(C *F, const char *name)
{
  const R d1 = TWO_PI / N1;
  const R d2 = TWO_PI / N2;

  R *h = (R *)Host;
  R *f = (R *)F;

  R (*func)(R, R);
  Z i, j;

       if(!strcmp(name, "zeros")) func = zeros;
  else if(!strcmp(name, "noise")) func = noise;
  else if(!strcmp(name, "decay")) func = decay;
  else if(!strcmp(name, "KH"   )) func = KH;
  else return NULL;

  for(i = 0; i < N1; ++i) {
    for(j = 0; j < N2; ++j) h[i * F2 + j] = func(d1 * i, d2 * j);
    for(     ; j < F2; ++j) h[i * F2 + j] = 0.0;
  }

  cudaMemcpy(f, h, sizeof(R) * N1 * F2, cudaMemcpyHostToDevice);
  return scale(forward(F, f), 1.0 / (N1 * N2));
}
