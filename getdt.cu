#include <stdio.h>
#include <math.h>
#include "ihd.h"

#ifndef CFL
#define CFL 0.707106781186547524 /* 1/sqrt(2) */
#endif

R getdt(R nu, R mu)
{
  const R n = MIN(N1, N2);
  R uu, adv, dff;

  getu(X, Y, W);
  reduce(&uu, NULL, inverse((R *)X, X), inverse((R *)Y, Y));

  adv = 10.0 / (sqrt(uu) * n);
  dff = 5.95 / (nu * n * n / 9.0 + mu);
  return CFL * MIN(adv, dff);
}
