#include <stdlib.h>
#include <cufft.h>
#include "ihd.h"

#define CONCATENATION(prefix, name) prefix ## name
#define CONCATE_MACRO(prefix, name) CONCATENATION(prefix, name)

#define FAIL(X, ...) (CUFFT_SUCCESS != CONCATE_MACRO(cufft, X)(__VA_ARGS__))

#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
#  define CUFFT_R2C    CUFFT_D2Z
#  define CUFFT_C2R    CUFFT_Z2D
#  define ExecR2C      ExecD2Z
#  define ExecC2R      ExecZ2D
#  define cufftReal    cufftDoubleReal
#  define cufftComplex cufftDoubleComplex
#endif

static cufftHandle r2c, c2r;

void mkplans(Z n1, Z n2)
{
  if(FAIL(Plan2d, &r2c, n1, n2, CUFFT_R2C) ||
     FAIL(Plan2d, &c2r, n1, n2, CUFFT_C2R) ||
     FAIL(SetCompatibilityMode, r2c, CUFFT_COMPATIBILITY_FFTW_PADDING) ||
     FAIL(SetCompatibilityMode, c2r, CUFFT_COMPATIBILITY_FFTW_PADDING)) {
    fprintf(stderr, "CUFFT ERROR :\tfail to create plan(s).\n");
    exit(-1);
  }
}

void rmplans(void)
{
  if(FAIL(Destroy, r2c) ||
     FAIL(Destroy, c2r)) {
    fprintf(stderr, "CUFFT ERROR :\tfail to destroy plan(s).\n");
    exit(-1);
  }
}

C *forward(C *F, R *f)
{
  if(FAIL(ExecR2C, r2c, (cufftReal *)f, (cufftComplex *)F)) {
    fprintf(stderr, "CUFFT ERROR :\tfail to perform forward transform.\n");
    exit(-1);
  }
  return F;
}

R *inverse(R *f, C *F)
{
  if(FAIL(ExecC2R, c2r, (cufftComplex *)F, (cufftReal *)f)) {
    fprintf(stderr, "CUFFT ERROR :\tfail to perform inverse transform.\n");
    exit(-1);
  }
  return f;
}
