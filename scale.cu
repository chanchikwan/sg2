#include "ihd.h"

static __global__ void _zero(C *f, const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    f[h].r = K(0.0);
    f[h].i = K(0.0);
  }
}

static __global__ void _scal(C *f, const R s, const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    f[h].r *= s;
    f[h].i *= s;
  }
}

C *scale(C *f, R s)
{
  if(s == 0.0) _zero<<<Hsz, Bsz>>>(f,    N1, H2);
  else         _scal<<<Hsz, Bsz>>>(f, s, N1, H2);
  return f;
}

static __global__ void _zero(R *f, const Z n1, const Z n2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j;

  if(i < n1 && j < n2)
    f[h] = 0.0;
}

static __global__ void _scal(R *f, const R s, const Z n1, const Z n2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j;

  if(i < n1 && j < n2)
    f[h] *= s;
}

R *scale(R *f, R s)
{
  if(s == 0.0) _zero<<<Gsz, Bsz>>>(f,    N1, N2);
  else         _scal<<<Gsz, Bsz>>>(f, s, N1, N2);
  return f;
}
