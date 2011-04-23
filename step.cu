#include "ihd.h"

static __global__ void _evol_diff(C *f, const C *b, const R KK,
                                        const R nu, const R mu,
                                        const R im, const R ex,
                                        const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const C g  = f[h];
    const C c  = b[h];
    const R kx = i < n1 / 2 ? i : i - n1;
    const R ky = j;
    const R kk = kx * kx + ky * ky;

    if(kk < KK) {
      const R imkk = im     * (nu * kk + mu);
      const R temp = K(1.0) / (K(1.0) + imkk);
      const R impl = temp   * (K(1.0) - imkk);
      const R expl = temp   * ex;

      f[h].r = impl * g.r + expl * c.r;
      f[h].i = impl * g.i + expl * c.i;
    } else {
      f[h].r = K(0.0);
      f[h].i = K(0.0);
    }
  }
}

void step(R nu, R mu, R fi, R ki, R dt)
{
  const R K = 0.99 + (MIN(N1, N2) - 1) / 3;

  const R alpha[] = {0.0,             0.1496590219993, 0.3704009573644,
                     0.6222557631345, 0.9582821306748, 1.0};
  const R beta[]  = {0.0,            -0.4178904745,   -1.192151694643,
                     -1.697784692471, -1.514183444257};
  const R gamma[] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                     0.6994504559488, 0.1530572479681};

  int i;
  for(i = 0; i < 5; ++i) {
    const R im = dt * 0.5 * (alpha[i+1] - alpha[i]);
    const R ex = dt * gamma[i] / (N1 * N2);

    if(fi * ki > 0.0)
      scale(w, beta[i]);
    else
      force(w, beta[i], fi, ki); /* scaling and Kolmogorov forcing */

    dx_dd_dy(X, Y, W); add_pro(w, inverse((R *)X, X), inverse((R *)Y, Y));
    dy_dd_dx(Y, X, W); sub_pro(w, inverse((R *)Y, Y), inverse((R *)X, X));

    forward(X, w); /* X here is just a buffer */

    _evol_diff<<<Hsz, Bsz>>>(W, (const C *)X, K * K, nu, mu, im, ex, N1, H2);
  }

  if(fi * ki > 0)
    force(W, dt, fi, ki); /* 1st-order Euler update for random forcing */
}
