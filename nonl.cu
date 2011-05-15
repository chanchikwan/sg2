#include "ihd.h"

/* Adding the 1st term in the Jacobian */
static __global__ void _add_pro(R *f, const R *x, const R *y,
                                      const Z n1, const Z n2, const Z N2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j; /* will be transformed out-of-place, no padding */
  const Z H = i * N2 + j;

  if(i < n1 && j < n2) f[h] += x[H] * y[H];
}

R *add_pro(R *f, R *y, R *x)
{
  _add_pro<<<Gsz, Bsz>>>(f, y, x, N1, N2, F2);
  return f;
}

/* Adding the 2nd term in the Jacobian */
static __global__ void _sub_pro(R *f, const R *y, const R *x,
                                      const Z n1, const Z n2, const Z N2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j; /* will be transformed out-of-place, no padding */
  const Z H = i * N2 + j;

  if(i < n1 && j < n2) f[h] -= y[H] * x[H];
}

R *sub_pro(R *f, R *y, R *x)
{
  _sub_pro<<<Gsz, Bsz>>>(f, y, x, N1, N2, F2);
  return f;
}
