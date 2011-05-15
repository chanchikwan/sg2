#include <stdio.h>
#include "ihd.h"

R getdt(R nu, R mu)
{
  const double n   = MIN(N1, N2);
  const double adv = 10.0 / (diag() * n);
  const double dff = 5.95 / (nu * n * n / 9.0 + mu);

  return CFL * MIN(adv, dff);
}
