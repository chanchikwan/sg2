#include <stdio.h>
#include "ihd.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

R getdt(R u, R nu, R mu)
{
  const double n   = MIN(N1, N2);
  const double adv = 10.0 / (u  * n);
  const double dff = 5.95 / (nu * n * n / 9.0 + mu);

  return MIN(adv, dff);
}
