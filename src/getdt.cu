/* Copyright (C) 2010-2011 Chi-kwan Chan
   Copyright (C) 2010-2011 NORDITA

   This file is part of sg2.

   Sg2 is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Sg2 is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with sg2. If not, see <http://www.gnu.org/licenses/>. */

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
