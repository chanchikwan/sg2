#include <stdio.h>
#include "ihd.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

R getdt(R u, R nu)
{
  const double n   = MIN(N1, N2);
  const double adv = 10.02  / (u  * n);
  const double dff = 53.567 / (nu * n * n);

  return MIN(adv, dff);
}
