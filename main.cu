#include <stdlib.h>
#include <stdio.h>
#include "ihd.h"

#define NOVAL (i+1 == argc) || (argv[i+1][0] == '-')
#define BREAK if(NOVAL) break
#define FLAG(x) case x:
#define PARA(x) case x: if(NOVAL) goto ignore; /* real programmers can write
                                                  FORTRAN in any language */
int main(int argc, char *argv[])
{
  const char *input = "zeros";

  R nu = 5.0e-3, mu = 0.0e+0;
  R fi = 1.0e+0, ki = 1.0e+1;
  R tt = 1000.0;

  Z n0 = 1000, n1 = 512, n2 = 512;
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
      PARA('r')Seed= atoi(argv[++i]); break;
      PARA('d') id = atoi(argv[++i]); break;
      PARA('n') nu = atof(argv[++i]); break;
      PARA('m') mu = atof(argv[++i]); break;
      PARA('f') ki = atof(argv[++i]); BREAK;
      fi = ki;  ki = atof(argv[++i]); break;
      PARA('t') tt = atof(argv[++i]); break;
      PARA('s') n0 = atoi(argv[++i]); BREAK;
           n2 = n1 = atoi(argv[++i]); BREAK;
                n2 = atoi(argv[++i]); break;
      PARA('o') setprefix(argv[++i]); break;
      default : ignore : printf("Ignore parameter \"%s\"\n", argv[i]);
    }
  }

  /* Pick a device */
  cudaGetDeviceCount(&i);
  printf("  Device %d/%d  :\t ", id, i);
  fflush(stdout);
  if(id < i) {
    cudaDeviceProp prop;
    if(cudaSuccess == cudaGetDeviceProperties(&prop, id) &&
       cudaSuccess == cudaSetDevice(id))
      printf("\b\"%s\" with %g MiB of memory\n",
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
  printf("  Dissipation :\t nu = %g,\tmu = %g\n", nu, mu);
  printf("  Forcing     :\t fi = %g,\tki = %g\n", fi, ki);
  printf("  Time        :\t tt = %g,\tnt = %d\n", tt, n0);
  printf("  Resolution  :\t n1 = %d,\tn2 = %d\n", n1, n2);
  setup(n1, n2);

  /* Load input file */
  if(exist(input)) {
    printf("  Input file  :\t ");
    if(load(W, input)) {
      i = frame(input);
      printf("loaded \"%s\"\n", input);
    } else {
      fflush(stdout);
      fprintf(stderr, "invalid input file \"%s\", QUIT\n", input);
      exit(-1);
    }
  }
  /* Initialize the fields */
  else {
    printf("  Initialize  :\t ");
    if(init(W, input)) {
      dump(name(i = 0), W);
      printf("\b\"%s\"\n", input);
    } else {
      fflush(stdout);
      fprintf(stderr, "invalid initial condition \"%s\", QUIT\n", input);
      exit(-1);
    }
  }

  /* Really solve the problem */
  return solve(nu, mu, fi, ki, tt / n0, i, n0);
}
