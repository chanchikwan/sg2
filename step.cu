#include <stdio.h>
#include <string.h>
#include "ihd.h"

static void lsRKCN4(R nu, R mu, R fi, R ki, R dt)
{
  const R alpha[] = {0.0,             0.1496590219993, 0.3704009573644,
                     0.6222557631345, 0.9582821306748, 1.0};
  const R beta [] = {0.0,            -0.4178904745,   -1.192151694643,
                    -1.697784692471, -1.514183444257};
  const R gamma[] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                     0.6994504559488, 0.1530572479681};
  lsRKCNn(5, alpha, beta, gamma, nu, mu, fi, ki, dt);
}

static void lsRKCN3(R nu, R mu, R fi, R ki, R dt)
{
  const R alpha[] = {0.0, 1.0/3.0, 3.0/4.0, 1.0};
  const R beta [] = {0.0, -5.0/9.0, -153.0/128.0};
  const R gamma[] = {1.0/3.0, 15.0/16.0, 8.0/15.0};
  lsRKCNn(3, alpha, beta, gamma, nu, mu, fi, ki, dt);
}

void setrk(const char *rk, Z k)
{
  Z subs;

  if(!strcmp("rk4", rk)) {
    subs = 5;
    Step = lsRKCN4;
  } else if(!strcmp("rk3", rk)) {
    subs = 3;
    Step = lsRKCN3;
  } else {
    fflush(stdout);
    fprintf(stderr, " is not implemented, QUIT\n");
    exit(-1);
  }

  Flop = N1 * N2 *        ( 4.0 +  5.0 * (log2((double)N1) + log2((double)N2)))
       + N1 * N2 * subs * (21.5 + 12.5 * (log2((double)N1) + log2((double)N2)))
       + N1 * N2 * subs * (k ? 8 : 0);
}
