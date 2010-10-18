#include <cufft.h>
#include "ihd.h"

#define FAIL(X, ...) (CUFFT_SUCCESS != cufft##X(__VA_ARGS__))

static cufftHandle r2c, c2r;

void mkplans(Z n1, Z n2)
{
  if(FAIL(Plan2d, &r2c, n1, n2, CUFFT_R2C) ||
     FAIL(Plan2d, &c2r, n1, n2, CUFFT_C2R)) {
    fprintf(stderr, "cufft error: fail to create plan(s).\n");
    exit(-1);
  }
}

void rmplans(void)
{
  if(FAIL(Destroy, r2c) ||
     FAIL(Destroy, c2r)) {
    fprintf(stderr, "cufft error: fail to destroy plan(s).\n");
    exit(-1);
  }
}

C *forward(C *F, R *f)
{
  if(FAIL(ExecR2C, r2c, (cufftReal *)f, (cufftComplex *)F)) {
    fprintf(stderr, "cufft error: fail to perform forward transform.\n");
    exit(-1);
  }
  return F;
}

R *inverse(R *f, C *F)
{
  if(FAIL(ExecC2R, c2r, (cufftComplex *)F, (cufftReal *)f)) {
    fprintf(stderr, "cufft error: fail to perform inverse transform.\n");
    exit(-1);
  }
  return f;
}
