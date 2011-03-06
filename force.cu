#include <stdlib.h>
#include "ihd.h"

/* We turn on Kolmogorov forcing when fi * ki < 0.  Therefore, there
   is an extra minus sign in line 19 although we are using

     f_K = fi ki cos(ki x)

   as the forcing. */
static __global__ void _force(R *f, const R sl, const R fi, const R ki,
                                    const Z n1, const Z n2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j;

  if(i < n1 && j < n2) {
    const R dx = K(1.0) / n1;
    f[h] = sl * f[h] - fi * ki * cos(TWO_PI * ki * i * dx);
  }
}

R *force(R *f, R sl, R fi, R ki)
{
  _force<<<Gsz, Bsz>>>(f, sl, fi, ki, N1, N2);
  return f;
}

/* When ky != 0, the random forcing is straightforward. */
static __global__ void _force1(C *f, const R fx, const R fy,
                                     const Z kx, const Z ky,
                                     const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const Z k = i < n1 / 2 ? i : i - n1;
    if(k == kx && (j == ky || j == -ky)) {
      f[h].r += fx;
      f[h].i += fy;
    }
  }
}

/* After transforming along the y-direction, the ky == 0 column is
   real.  It is necessary to implement the Hermit symmetric by hand.
   The force is added to both the k == kx and k ==-kx modes. */
static __global__ void _force2(C *f, const R fx, const R fy,
                                     const Z kx, const Z ky,
                                     const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const Z k = i < n1 / 2 ? i : i - n1;
    if(k == kx && j == ky) {
      f[h].r += fx;
      f[h].i += fy;
    }
    if(k ==-kx && j == ky) {
      f[h].r += fx;
      f[h].i -= fy;
    }
  }
}

C *force(C *f, R dt, R fi, R ki)
{
  const R fs = fi * ki * sqrt(dt) / 2.0; /* because of real-complex FFT */
  const R dp = TWO_PI / (RAND_MAX + 1.0);
  const R pm = dp * (Seed = rand());
  const R pk = dp * (Seed = rand());

  const R fx = fs * cos(pm);
  const R fy = fs * sin(pm);
  const Z kx = (Z)floor(ki * cos(pk) + 0.5);
  const Z ky = (Z)floor(ki * sin(pk) + 0.5);

  if(ky) _force1<<<Hsz, Bsz>>>(f, fx, fy, kx, ky, N1, H2);
  else   _force2<<<Hsz, Bsz>>>(f, fx, fy, kx, ky, N1, H2);

  return f;
}
