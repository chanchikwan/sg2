#include <cufft.h>
#include "sg2.h"

#define CONCATENATION(PREFIX, NAME) PREFIX ## NAME
#define CONCATE_MACRO(PREFIX, NAME) CONCATENATION(PREFIX, NAME)

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
     FAIL(SetCompatibilityMode, c2r, CUFFT_COMPATIBILITY_FFTW_PADDING))
    error("CUFFT ERROR :\tfail to create plan(s).\n");
}

void rmplans(void)
{
  if(FAIL(Destroy, r2c) ||
     FAIL(Destroy, c2r))
    error("CUFFT ERROR :\tfail to destroy plan(s).\n");
}

C *forward(C *F, R *f)
{
  if(FAIL(ExecR2C, r2c, (cufftReal *)f, (cufftComplex *)F))
    error("CUFFT ERROR :\tfail to perform forward transform.\n");
  return F;
}

R *inverse(R *f, C *F)
{
  if(FAIL(ExecC2R, c2r, (cufftComplex *)F, (cufftReal *)f))
    error("CUFFT ERROR :\tfail to perform inverse transform.\n");
  return f;
}
