#include "ihd.h"

/* Compute df/dx and dw/dy from w for the 1st term in the Jacobian */
static __global__ void _dx_dd_dy(C *x, C *y, const C *w,
                                 const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const R kx    = i < n1 / 2 ? i : i - n1;
    const R ky    = j;
    const R kx_kk = kx / (kx * kx + ky * ky + 1.0e-16f);
    const C u     = w[h];

    x[h].r = - kx_kk * u.i;
    x[h].i =   kx_kk * u.r;
    y[h].r = - ky    * u.i;
    y[h].i =   ky    * u.r;
  }
}

void dx_dd_dy(C *x, C *y, C *w)
{
  _dx_dd_dy<<<Hsz, Bsz>>>(x, y, w, N1, H2);
}

/* Compute df/dy and dw/dx from w for the 2nd term in the Jacobian */
static __global__ void _dy_dd_dx(C *y, C *x, const C *w,
                                 const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const R kx    = i < n1 / 2 ? i : i - n1;
    const R ky    = j;
    const R ky_kk = ky / (kx * kx + ky * ky + 1.0e-16f);
    const C u     = w[h];

    y[h].r = - ky_kk * u.i;
    y[h].i =   ky_kk * u.r;
    x[h].r = - kx    * u.i;
    x[h].i =   kx    * u.r;
  }
}

void dy_dd_dx(C *y, C *x, C *w)
{
  _dy_dd_dx<<<Hsz, Bsz>>>(y, x, w, N1, H2);
}
