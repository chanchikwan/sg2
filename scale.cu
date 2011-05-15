#include "ihd.h"

#define ZERO(T, op)                                             \
  static __global__ void _zero(T *f, const Z N1, const Z H2)    \
  {                                                             \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;          \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;          \
    const Z h = i * H2 + j;                                     \
                                                                \
    if(i < N1 && j < H2) {                                      \
      op;                                                       \
    }                                                           \
  }

#define SCAL(T, op)                                                     \
  static __global__ void _scal(T *f, const R s, const Z N1, const Z H2) \
  {                                                                     \
    const Z i = blockDim.y * blockIdx.y + threadIdx.y;                  \
    const Z j = blockDim.x * blockIdx.x + threadIdx.x;                  \
    const Z h = i * H2 + j;                                             \
                                                                        \
    if(i < N1 && j < H2) {                                              \
      op;                                                               \
    }                                                                   \
  }

ZERO(C, f[h].i = f[h].r = K(0.0))
SCAL(C, f[h].r *= s; f[h].i *=s)

C *scale(C *F, R s)
{
  if(s == 0.0) _zero<<<Hsz, Bsz>>>(F,    N1, H2);
  else         _scal<<<Hsz, Bsz>>>(F, s, N1, H2);
  return F;
}

ZERO(R, f[h] = K(0.0))
SCAL(R, f[h] *= s)

R *scale(R *f, R s)
{
  if(s == 0.0) _zero<<<Gsz, Bsz>>>(f,    N1, N2);
  else         _scal<<<Gsz, Bsz>>>(f, s, N1, N2);
  return f;
}
