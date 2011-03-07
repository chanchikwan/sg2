#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ihd.h"

#define HAS_ARG (i+1 < argc && argv[i+1][0] != '-')

R noise(R x, R y)
{
  return 0.5 - (R)(Seed = rand()) / (RAND_MAX + 1.0);
}

R decay(R x, R y)
{
  return 1024.0 * noise(x, y); /* so u = curl(w) ~ 1 */
}

R KH(R x, R y)
{
  return noise(x, y) + (fabs(x - 0.0 ) < 1.0e-6 ? -512.0 : 0.0)
                     + (fabs(x - M_PI) < 1.0e-6 ?  512.0 : 0.0);
}

int main(int argc, char *argv[])
{
  const char  rotor[] = "-/|\\";
  const char *input   = NULL;

  R nu = 1.0e-5, mu = 1.0e-2;
  R fi = 5.0e-2, ki = 1.0e+2;
  R tt = 1024.0, fo;

  Z n0 = 1024, n1 = 1024, n2 = 1024, i;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  for(i = 1; i < argc; ++i) if(!strcmp(argv[i], "--help")) usage(0);
  printf("2D spectral hydrodynamic code with CUDA\n\n");

  for(i = 1; i < argc; ++i) {
    /* Arguments do not start with '-' are input file names */
    if(argv[i][0] != '-')
      input = argv[i];
    /* Arguments start with '-' are options */
    else switch(argv[i][1]) {
      case 'n': if(HAS_ARG) { nu = atof(argv[++i]); break; }
      case 'm': if(HAS_ARG) { mu = atof(argv[++i]); break; }
      case 'f': if(HAS_ARG) { fi = atof(argv[++i]); break; }
      case 'k': if(HAS_ARG) { ki = atof(argv[++i]); break; }
      case 't': if(HAS_ARG) { tt = atof(argv[++i]); break; }
      case 's': if(HAS_ARG) { n0 = atoi(argv[++i]);
                if(HAS_ARG)   n1 = atoi(argv[++i]);
                if(HAS_ARG)   n2 = atoi(argv[++i]); break; }
      case 'o': if(HAS_ARG) { setprefix(argv[++i]); break; }
      default : printf("Ignore \"%s\"\n", argv[i]);
    }
  }

  printf("Dissipation :\tnu = %g,\tmu = %g\n", nu, mu);
  printf("Forcing     :\tfi = %g,\tki = %g\n", fi, ki);
  printf("Time        :\ttt = %g,\tnt = %d\n", tt, n0);
  printf("Resolution  :\tn1 = %d,\tn2 = %d\n", n1, n2);
  printf("Input/init  :\t\"%s\", ", input ? input : "none");

  setup(n1, n2);
  fo = 5 * n1 * n2 * (21.5 + (fi * ki > 0.0 ? 0 : 8) +
                      12.5 * (log2((double)n1) + log2((double)n2)));

  if(input && exist(input) && load(w, input)) {
    scale(forward(W, w), 1.0 / (n1 * n2));
    printf("LOADED\n");
    i = frame(input);
  } else {
    if(input) printf("FAILED TO LOAD, ");
    scale(forward(W, init(w, noise)), 1.0 / (n1 * n2));
    printf("initialized with noise\n");
    dump(name(i = 0), inverse(w, W));
  }

  printf("======================= Start simulation =======================\n");

  for(++i; i <= n0; ++i) {
    float ms;
    Z ns = (Z)ceil(tt / n0 / 0.9 / getdt(1.0, nu, mu)), j;
    R dt =         tt / n0 / ns;
    printf("%4d: %5.2f -> %5.2f, dt ~ %.0e:       ",
           i, dt * ns * (i-1), dt * ns * i, dt);

    cudaEventRecord(t0, 0);
    for(j = 0; j < ns; ++j) {
      printf("\b\b\b\b\b\b%c %4d", rotor[j%4], j+1);
      fflush(stdout);
      step(nu, mu, fi, ki, dt);
    }
    cudaEventRecord(t1, 0);

    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1); ms /= ns;
    printf("\b\b\b\b\b\b%.3f ms/cycle ~ %.3f GFLOPS\n",
           ms, 1e-6 * fo / ms);

    dump(name(i), inverse(w, W));
  }

  cudaEventDestroy(t1);
  cudaEventDestroy(t0);

  return 0;
}
