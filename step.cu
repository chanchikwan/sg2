#include "ihd.h"

#define TIDE_Y 32

static __global__ void _adv(C *f, R nu, R dt, Z n1, Z h2)
{
  Z i = blockDim.x * blockIdx.x + threadIdx.x;
  Z j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i < n1 && j < h2) {
    Z h = i * h2 + j;
    C F = f[h];

    R kx = i < n1 / 2 ? i : i - n1;
    R ky = j;
    R kk = kx * kx + ky * ky;

    R dr = 1.0f + dt * nu * kk;
    R di = dt * (kx + ky);
    R id = 1.0f / (dr * dr + di * di);

    f[h].r = id * (dr * F.r + di * F.i);
    f[h].i = id * (dr * F.i - di * F.r);
  }
}

void step(R nu, R dt)
{
  uint3 bsz = make_uint3(M / TIDE_Y, TIDE_Y, 1);
  uint3 gsz = make_uint3((N1 - 1) / bsz.x + 1,
                         (N2 / 2) / bsz.y + 1, 1);

  _adv<<<gsz, bsz>>>(W, nu, dt, N1, H2);
}
