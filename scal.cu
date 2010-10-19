#include "ihd.h"

__global__ void _scal(C *f, R s, Z n)
{
  Z i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n) {
    f[i].r *= s;
    f[i].i *= s;
  }
}

C *scal(C *f, R s)
{
  Z n = N1 * H2;
  _scal<<<(n - 1) / M + 1, M>>>(f, s, n);
  return f;
}

__global__ void _scal(R *f, R s, Z n)
{
  Z i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n) f[i] *= s;
}

R *scal(R *f, R s)
{
  Z n = N1 * N2;
  _scal<<<(n - 1) / M + 1, M>>>(f, s, n);
  return f;
}
