#include <stdio.h>
#include "ihd.h"

#define TIDE 512

static __global__ void _reduce(C *out, const R *x, const R *y,
                                       const Z n1, const Z n2, const Z N2)
{
  __shared__ R max[TIDE], sum[TIDE];

  const Z t = threadIdx.x;
  const Z j = blockDim.x * blockIdx.x + t;
  Z i;

  max[t] = 0.0;
  sum[t] = 0.0;

  if(j < n2) for(i = 0; i < n1; ++i) {
    const Z h  = i * N2 + j;
    const R ux = x[h];
    const R uy = y[h];
    const R uu = ux * ux + uy * uy;
    if(max[t] < uu) max[t] = uu;
    sum[t] += uu;
  }
  __syncthreads();

  for(i = blockDim.x / 2; i > 0; i /= 2) {
    if(t < i) {
      if(max[t] < max[t + i]) max[t] = max[t + i];
      sum[t] += sum[t + i];
    }
    __syncthreads();
  }

  if(t == 0) {
    out[blockIdx.x].r = max[0];
    out[blockIdx.x].i = sum[0];
  }
}

static FILE *file = NULL;

static void close(void)
{
  if(file) fclose(file);
}

R diag(void)
{
  const Z bsz = TIDE;
  const Z gsz = (N2 - 1) / bsz + 1;

  Z i;
  R max = 0.0, sum = 0.0;

  dx_dd_dy_dd(X, Y, W);
  inverse((R *)X, X);
  inverse((R *)Y, Y);
  _reduce<<<gsz, bsz>>>((C *)w, (R *)X, (R *)Y, N1, N2, F2);

  cudaMemcpy(Host, w, sizeof(C) * gsz, cudaMemcpyDeviceToHost);

  for(i = 0; i < gsz; ++i) {
    if(max < Host[i].r) max = Host[i].r;
    sum += Host[i].i;
  }

  if(!file) {
    atexit(close);
    file = fopen("log", "w");
  }

  fprintf(file, "%g\n", 0.5 * sum / (N1 * N2));
  fflush(file);

  return sqrt(max);
}
