#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

#define HAS_ARG (i+1 < argc && argv[i+1][0] != '-')

int main(int argc, char *argv[])
{
  const char *input = "zeros";

  R nu = 1.0e-5, mu = 1.0e-2;
  R fi = 5.0e-2, ki = 1.0e+2;
  R tt = 1024.0;

  Z n0 = 1024, n1 = 1024, n2 = 1024;
  Z id = 0, i;

  /* If "--help" is an argument, print usage and exit */
  for(i = 1; i < argc; ++i) if(!strcmp(argv[i], "--help")) usage(0);
  printf("Spectral Galerkin Incompressible Hydrodynamic in 2D (with CUDA)\n");

  /* Home made argument parser */
  for(i = 1; i < argc; ++i) {
    /* Arguments do not start with '-' are input file names */
    if(argv[i][0] != '-') input = argv[i];
    /* Arguments start with '-' are options */
    else switch(argv[i][1]) {
      case 'd': if(HAS_ARG) { id = atoi(argv[++i]); break; }
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

  /* Pick a device */
  cudaGetDeviceCount(&i);
  printf("Device %d/%d  :\t", id, i);
  fflush(stdout);
  if(id < i) {
    cudaDeviceProp prop;
    if(cudaSuccess == cudaGetDeviceProperties(&prop, id) &&
       cudaSuccess == cudaSetDevice(id))
      printf("\"%s\" with %g MiB of memory\n",
             prop.name, prop.totalGlobalMem / 1024.0 / 1024.0);
    else {
      fprintf(stderr, "fail to access device, QUIT\n");
      exit(-1);
    }
  } else {
    fprintf(stderr, "device id is too large, QUIT\n");
    exit(-1);
  }

  /* Print simulation setup */
  printf("Dissipation :\tnu = %g,\tmu = %g\n", nu, mu);
  printf("Forcing     :\tfi = %g,\tki = %g\n", fi, ki);
  printf("Time        :\ttt = %g,\tnt = %d\n", tt, n0);
  printf("Resolution  :\tn1 = %d,\tn2 = %d\n", n1, n2);
  setup(n1, n2);

  /* Load input file or initialize the fields */
  if(exist(input)) {
    printf("Input file  :\t");
    if(load(w, input)) {
      scale(forward(W, w), 1.0 / (n1 * n2));
      printf("loaded \"%s\"\n", input);
      i = frame(input);
    } else {
      fflush(stdout);
      fprintf(stderr, "invalid input file \"%s\", QUIT\n", input);
      exit(-1);
    }
  } else {
    printf("Initialize  :\t");
    if(init(w, input)) {
      scale(forward(W, w), 1.0 / (n1 * n2));
      printf("\"%s\"\n", input);
      dump(name(i = 0), inverse(w, W));
    } else {
      fflush(stdout);
      fprintf(stderr, "invalid initial condition \"%s\", QUIT\n", input);
      exit(-1);
    }
  }

  /* Really solve the problem */
  return solve(nu, mu, fi, ki, tt, i, n0);
}
