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
   along with sg2.  If not, see <http://www.gnu.org/licenses/>. */

#include <string.h>
#include <math.h>
#include "sg2.h"

static Z subs = 0, Kforce = 0;

static void lsRKCN3(R nu, R mu, R bt, R fi, R ki, R dt)
{
  const R alpha[] = {0.0, 1.0/3.0, 3.0/4.0, 1.0};
  const R beta [] = {0.0, -5.0/9.0, -153.0/128.0};
  const R gamma[] = {1.0/3.0, 15.0/16.0, 8.0/15.0};
  lsRKCNn(3, alpha, beta, gamma, nu, mu, bt, fi, ki, dt);
}

static void lsRKCN4(R nu, R mu, R bt, R fi, R ki, R dt)
{
  const R alpha[] = {0.0,             0.1496590219993, 0.3704009573644,
                     0.6222557631345, 0.9582821306748, 1.0};
  const R beta [] = {0.0,            -0.4178904745,   -1.192151694643,
                     -1.697784692471, -1.514183444257};
  const R gamma[] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                     0.6994504559488, 0.1530572479681};
  lsRKCNn(5, alpha, beta, gamma, nu, mu, bt, fi, ki, dt);
}

void setrk(const char *rk)
{
  if(!strcmp("rk3", rk))
    subs = 3;
  else if(!strcmp("rk4", rk))
    subs = 5;
  else error(" is not implemented, QUIT\n");
}

void step(R nu, R mu, R bt, R fi, R ki, R dt)
{
  switch(subs) {
  case 3 : lsRKCN3(nu, mu, bt, fi, ki, dt); break;
  case 5 : lsRKCN4(nu, mu, bt, fi, ki, dt); break;
  default: error("substep number doesn't match any stepper, QUIT"); break;
  }
  Kforce = fi * ki < 0.0;
}

R flop(void)
{
  return N1 * N2 *        ( 4.0 +  5.0 * (log2((R)N1) + log2((R)N2)))
       + N1 * N2 * subs * (26.5 + 12.5 * (log2((R)N1) + log2((R)N2)))
       + N1 * N2 * subs * (Kforce ? 8 : 0);
  /* Float operation count includes reduction + RK4 + forcing, the
     computation of the nonlinear term in raw-dump is not included. */
}
