/* Copyright (C) 2010-2011 Chi-kwan Chan
   Copyright (C) 2010-2011 NORDITA

   This file is part of sg2.

   Sg2 is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Sg2 is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with sg2. If not, see <http://www.gnu.org/licenses/>. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sg2.h"

#define NOVAL (i+1 == argc) || (argv[i+1][0] == '-')
#define BREAK if(NOVAL) break
#define FLAG(X) case X:
#define PARA(X) case X: if(NOVAL) goto ignore; /* real programmers can write
                                                  FORTRAN in any language */
int main(int argc, char *argv[])
{
  const char *input = "zeros";
  const char *rk    = "rk3";

  R nu = 5.0e-3, mu = 0.0e+0;
  R fi = 1.0e+0, ki = 1.0e+1;
  R tt = 1000.0, dt = 0.0e+0;

  Z n0 = 1000, n1 = 512, n2 = 512;
  Z id = 0, i;

  /* If "--help" is an argument, print usage and exit */
  for(i = 1; i < argc; ++i) if(!strcmp(argv[i], "--help")) usage(0);
  printf("Spectral Galerkin Incompressible Hydrodynamic in 2D (with CUDA)\n");

  /* Home made argument parser */
  for(i = 1; i < argc; ++i) {
    /* Arguments do not start with '-' are random seed or input file names */
    if(argv[i][0] != '-') {
      if(setseed(argv[i]) < 0) input = argv[i];
    }
    /* Arguments start with '-' are options */
    else switch(argv[i][1]) {
      PARA('d') id = atoi(argv[++i]); break;
      PARA('n') nu = atof(argv[++i]); break;
      PARA('m') mu = atof(argv[++i]); break;
      PARA('f') ki = atof(argv[++i]); BREAK;
      fi = ki;  ki = atof(argv[++i]); break;
      FLAG('r') rk =      argv[i]+1 ; break;
      PARA('t') tt = atof(argv[++i]); BREAK;
                dt = atof(argv[++i]); break;
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
    else error("fail to access device, QUIT\n");
  } else error("device id is too large, QUIT\n");

  /* Print simulation setup */
  printf("  Integrator  :\t\"%s\"", rk);
  setrk(rk);

  if(tt < dt) { /* need to reinterpret inputs */
    R cfl = tt;
    tt = dt; dt = 0.0;
    printf(" with adaptive time step, CFL = %g\n", cfl);
    setdt(cfl, DT_MIN); /* TODO: choose minimum dt at runtime */
  } else if(dt == 0.0) {
    printf(" with adaptive time step, CFL = %g\n", CFL);
  } else {
    printf(" with fixed time step %g\n", dt);
    setdt(0.0, dt);
  }

  printf("  Dissipation :\t nu = %g,\tmu = %g\n", nu, mu);
  printf("  Forcing     :\t fi = %g,\tki = %g\n", fi, ki);
  printf("  Time        :\t tt = %g,\tnt = %d\n", tt, n0);
  printf("  Resolution  :\t n1 = %d,\tn2 = %d\n", n1, n2);
  setup(n1, n2);

  /* Load input file */
  if(valid(input)) {
    printf("  Input file  :\t ");
    if(load(W, input)) {
      i = frame(input);
      printf("loaded \"%s\"\n", input);
    } else
      error("invalid input file \"%s\", QUIT\n", input);
  }
  /* Initialize the fields */
  else {
    printf("  Initialize  :\t ");
    if(init(W, input)) {
      dump(name(i = 0), W);
      printf("\b\"%s\"\n", input);
    } else
      error("invalid initial condition \"%s\", QUIT\n", input);
  }

  /* Really solve the problem */
  return solve(nu, mu, fi, ki, tt / n0, i, n0);
}
