#include "ihd.h"

static __global__ void _adv(C *f, const R nu, const R dt,
                                  const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C g = f[h];

    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;
    const R kk = kx * kx + ky * ky;

    const R dr = 1.0f + dt * nu * kk;
    const R di = dt * (kx + ky);
    const R id = 1.0f / (dr * dr + di * di);

    f[h].r = id * (dr * g.r + di * g.i);
    f[h].i = id * (dr * g.i - di * g.r);
  }
}

void step(R nu, R dt)
{
  _adv<<<Hsz, Bsz>>>(W, nu, dt, N1, H2);
}
