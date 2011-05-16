#include "ihd.h"

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

void step(R nu, R mu, R fi, R ki, R dt)
{
  const R K = 0.99 + (MIN(N1, N2) - 1) / 3;

#if SUBS == 5
  const R alpha[] = {0.0,             0.1496590219993, 0.3704009573644,
                     0.6222557631345, 0.9582821306748, 1.0};
  const R beta [] = {0.0,            -0.4178904745,   -1.192151694643,
                    -1.697784692471, -1.514183444257};
  const R gamma[] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                     0.6994504559488, 0.1530572479681};
#elif SUBS == 3
  const R alpha[] = {0.0, 1.0/3.0, 3.0/4.0, 1.0};
  const R beta [] = {0.0, -5.0/9.0, -153.0/128.0};
  const R gamma[] = {1.0/3.0, 15.0/16.0, 8.0/15.0};
#elif
#error "Number of substeps does not match an implemented algorithm"
#endif

  int i;
  for(i = 0; i < SUBS; ++i) {
    const R im = dt * 0.5 * (alpha[i+1] - alpha[i]);
    const R ex = dt * gamma[i] / (N1 * N2);

    if(fi * ki > 0.0)
      scale(w, beta[i]);
    else
      force(w, beta[i], fi, ki); /* scaling and Kolmogorov forcing */

    jacobi1(X, Y, W); add_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));
    jacobi2(X, Y, W); sub_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));

    forward(X, w); /* X here is just a buffer */

    _evol_diff<<<Hsz, Bsz>>>(W, (const C *)X, K * K, nu, mu, im, ex, N1, H2);
  }

  if(fi * ki > 0)
    force(W, dt, fi, ki); /* 1st-order Euler update for random forcing */
}
