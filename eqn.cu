#include "sg2.h"

static __global__ void _evol_diff(C *W, const C *F, const R KK,
                                        const R nu, const R mu,
                                        const R im, const R ex,
                                        const Z N1, const Z H2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * H2 + j;

  if(i < N1 && j < H2) {
    const C Wh = W[h];
    const C Fh = F[h];
    const R kx = i < N1 / 2 ? i : i - N1;
    const R ky = j;
    const R kk = kx * kx + ky * ky;

    if(kk < KK) {
      const R imkk = im     * ( mu + nu * kk);
      const R temp = K(1.0) / (K(1.0) + imkk);
      const R impl = temp   * (K(1.0) - imkk);
      const R expl = temp   * ex;

      W[h].r = impl * Wh.r + expl * Fh.r;
      W[h].i = impl * Wh.i + expl * Fh.i;
    } else {
      W[h].r = K(0.0);
      W[h].i = K(0.0);
    }
  }
}

void lsRKCNn(const Z n, const R *alpha, const R *beta, const R *gamma,
             R nu, R mu, R fi, R ki, R dt)
{
  const R K = 0.99 + (MIN(N1, N2) - 1) / 3;

  Z i;
  for(i = 0; i < n; ++i) {
    const R im = dt * 0.5 * (alpha[i+1] - alpha[i]);
    const R ex = dt * gamma[i] / (N1 * N2);

    if(fi * ki < 0.0)
      force(w, beta[i], fi, ki); /* scaling and Kolmogorov forcing */
    else
      scale(w, beta[i]);

    jacobi1(X, Y, W); add_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));
    jacobi2(X, Y, W); sub_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));

    forward(X, w); /* X here is just a buffer */

    _evol_diff<<<Hsz, Bsz>>>(W, (const C *)X, K * K, nu, mu, im, ex, N1, H2);
  }

  if(fi * ki > 0)
    force(W, dt, fi, ki); /* 1st-order Euler update for random forcing */
}
