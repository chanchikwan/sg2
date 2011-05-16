#include <stdio.h>
#include <math.h>
#include "ihd.h"

R getdt(R nu, R mu, R fi)
{
  const R n = MIN(N1, N2);
  R uu, adv, dff, frc;

  cudaMemcpy(Host, W, sizeof(R), cudaMemcpyDeviceToHost);
  if(Host[0].r != Host[0].r) return 0.0; /* spectrum contains NAN */

  getu(X, Y, W);
  reduce(&uu, NULL, inverse((R *)X, X), inverse((R *)Y, Y));

  adv = 10.0 / (sqrt(uu) * n);
  dff = 5.95 / (nu * n * n / 9.0 + mu);
  frc = pow(10.0 / fabs(fi * n), 2.0 / 3.0);

  return CFL * MIN(adv, MIN(dff, frc));
}
