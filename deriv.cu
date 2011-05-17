#include "sg2.h"

#define KERN(NAME, OP)                                  \
  static __global__ void NAME(C *X, C *Y, const C *W,   \
                              const Z N1, const Z H2)   \
  {                                                     \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;  \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;  \
    const Z h = i * H2 + j;                             \
                                                        \
    if(i < N1 && j < H2) {                              \
      const C Wh = W[h];                                \
      R kx = i < N1 / 2 ? i : i - N1;                   \
      R ky = j;                                         \
      if(h) {                                           \
        const R ikk = K(1.0) / (kx * kx + ky * ky);     \
        OP;                                             \
      }                                                 \
                                                        \
      X[h].r = - kx * Wh.i;                             \
      X[h].i =   kx * Wh.r;                             \
      Y[h].r = - ky * Wh.i;                             \
      Y[h].i =   ky * Wh.r;                             \
    }                                                   \
  }

/* Compute dF/dx and dF/dy from W */
KERN(_Fx_Fy, kx *= ikk; ky *= ikk)

void getu(C *Fx, C *Fy, const C *W)
{
  _Fx_Fy<<<Hsz, Bsz>>>(Fx, Fy, W, N1, H2);
}

/* Compute dF/dx and dW/dy from W for the 1st term in the Jacobian */
KERN(_Fx_Wy, kx *= ikk)

void jacobi1(C *Fx, C *Wy, const C *W)
{
  _Fx_Wy<<<Hsz, Bsz>>>(Fx, Wy, W, N1, H2);
}

/* Compute dW/dx and dF/dy from W for the 2nd term in the Jacobian */
KERN(_Wx_Fy, ky *= ikk)

void jacobi2(C *Wx, C *Fy, const C *W)
{
  _Wx_Fy<<<Hsz, Bsz>>>(Wx, Fy, W, N1, H2);
}
