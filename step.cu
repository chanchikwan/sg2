#include "ihd.h"

static __global__ void _advection(C *b, const C *f, const R beta,
                                        const R n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C c  = b[h];
    const C g  = f[h];
    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;
    const R uk = kx + ky;

    b[h].r = beta * c.r + uk * g.i;
    b[h].i = beta * c.i - uk * g.r;
  }
}

static __global__ void _evol_diff(C *f, const C *b, const R im, const R ex,
                                        const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C g  = f[h];
    const C c  = b[h];
    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;

    const R imkk = im * (kx * kx + ky * ky);
    const R temp = 1.0f / (1.0f + imkk);
    const R impl = temp * (1.0f - imkk);
    const R expl = temp * ex;

    f[h].r = impl * g.r + expl * c.r;
    f[h].i = impl * g.i + expl * c.i;
  }
}

void step(R nu, R dt)
{
  const R alpha[] = {0.0,             0.1496590219993, 0.3704009573644,
                     0.6222557631345, 0.9582821306748, 1.0};
  const R beta[]  = {0.0,            -0.4178904745,   -1.192151694643,
                     -1.697784692471, -1.514183444257};
  const R gamma[] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                     0.6994504559488, 0.1530572479681};
  int i;
  for(i = 0; i < 5; ++i) {
    const R im = dt * nu * 0.5f * (alpha[i+1] - alpha[i]);
    const R ex = dt * gamma[i];

    _advection<<<Hsz, Bsz>>>((C *)w, (const C *)W, beta[i], N1, H2);
    _evol_diff<<<Hsz, Bsz>>>((C *)W, (const C *)w, im, ex, N1, H2);
  }
}
