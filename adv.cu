#include "ihd.h"

/* Compute dF/dx and dF/dy from W */
static __global__ void _dx_dd_dy_dd(C *x, C *y, const C *w,
                                    const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C u = w[h];
    R lx = i < n1 / 2 ? i : i - n1;
    R ly = j;

    if(h) {
      const R ikk = K(1.0) / (lx * lx + ly * ly);
      lx *= ikk;
      ly *= ikk;
    }

    x[h].r = - lx * u.i;
    x[h].i =   lx * u.r;
    y[h].r = - ly * u.i;
    y[h].i =   ly * u.r;
  }
}

void dx_dd_dy_dd(C *x, C *y, C *w)
{
  _dx_dd_dy_dd<<<Hsz, Bsz>>>(X, Y, W, N1, H2);
}

/* Compute dF/dx and dW/dy from W for the 1st term in the Jacobian */
static __global__ void _dx_dd_dy(C *x, C *y, const C *w,
                                 const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C u  = w[h];
    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;

    R lx = kx;
    if(h) lx *= K(1.0) / (kx * kx + ky * ky);

    x[h].r = - lx * u.i;
    x[h].i =   lx * u.r;
    y[h].r = - ky * u.i;
    y[h].i =   ky * u.r;
  }
}

void dx_dd_dy(C *x, C *y, C *w)
{
  _dx_dd_dy<<<Hsz, Bsz>>>(x, y, w, N1, H2);
}

/* Compute dF/dy and dW/dx from W for the 2nd term in the Jacobian */
static __global__ void _dy_dd_dx(C *y, C *x, const C *w,
                                 const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C u  = w[h];
    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;

    R ly = ky;
    if(h) ly *= K(1.0) / (kx * kx + ky * ky);

    y[h].r = - ly * u.i;
    y[h].i =   ly * u.r;
    x[h].r = - kx * u.i;
    x[h].i =   kx * u.r;
  }
}

void dy_dd_dx(C *y, C *x, C *w)
{
  _dy_dd_dx<<<Hsz, Bsz>>>(y, x, w, N1, H2);
}
