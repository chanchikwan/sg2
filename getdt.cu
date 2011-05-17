#include <math.h>
#include "sg2.h"

/* If cfl == 0.0, fix is the fixed step */
static R cfl = CFL;
static R fix = DT_MIN;

void setdt(R c, R f)
{
  cfl = c;
  fix = f;
}

R getdt(R nu, R mu, R fi)
{
  const R n = MIN(N1, N2);
  R uu, adv, dff, frc, dt;

  if(cfl == 0.0) return fix;

  cudaMemcpy(Host, W, sizeof(R), cudaMemcpyDeviceToHost);
  if(Host[0].r != Host[0].r) return 0.0; /* spectrum contains NAN */

  getu(X, Y, W);
  reduce(&uu, NULL, inverse((R *)X, X), inverse((R *)Y, Y));

  adv = 10.0 / (sqrt(uu) * n);
  dff = 5.95 / (nu * n * n / 9.0 + mu);
  frc = pow(10.0 / fabs(fi * n), 2.0 / 3.0);

  dt = cfl * MIN(adv, MIN(dff, frc));
  return dt < fix ? 0.0 : dt;
}
