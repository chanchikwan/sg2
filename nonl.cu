#include "sg2.h"

#define KERN(NAME, OP)                                          \
  static __global__ void NAME(R *f, const R *x, const R *y,     \
                        const Z N1, const Z N2, const Z F2)     \
  {                                                             \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;          \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;          \
    const Z h = i * N2 + j;                                     \
    const Z H = i * F2 + j;                                     \
                                                                \
    if(i < N1 && j < N2) f[h] OP x[H] * y[H];                   \
  }

/* Adding the 1st term in the Jacobian */
KERN(_add_pro, +=)

R *add_pro(R *f, const R *x, const R *y)
{
  _add_pro<<<Gsz, Bsz>>>(f, x, y, N1, N2, F2);
  return f;
}

/* Adding the 2nd term in the Jacobian */
KERN(_sub_pro, -=)

R *sub_pro(R *f, const R *x, const R *y)
{
  _sub_pro<<<Gsz, Bsz>>>(f, x, y, N1, N2, F2);
  return f;
}
