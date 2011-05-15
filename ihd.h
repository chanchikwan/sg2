#ifndef IHD_H
#define IHD_H

#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
#define K(X) (X)
#else
#define K(X) (X##f)
#endif

#define ONE_PI K(3.1415926535897932)
#define TWO_PI K(6.2831853071795865)

typedef int Z;
#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
typedef double R;
#else
typedef float R;
#endif
typedef struct {R r, i;} C;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* Ooops... global variables */
extern Z N1, N2, H2, F2, Seed;
extern uint3 Bsz, Gsz, Hsz;

extern R *w, CFL;
extern C *W, *X, *Y, *Host;

/* FFT wrapper in fft.cu */
void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

/* Miscellaneous in misc.cu, mostly used by main.cu */
void usage(int);
void setprefix(const char *);
int exist(const char *);
int frame(const char *);
const char *name(Z);
R diag(void);

/* Setup, helpers, and drivers */
void setup(Z, Z);
int  solve(R, R, R, R, R, Z, Z);
R    getdt(R, R);

C *init(C *, const char *);
C *load(C *, const char *);
C *dump(const char *, C *);

/* Computation kernels */
void getu   (C *, C *, const C *);
void jacobi1(C *, C *, const C *);
void jacobi2(C *, C *, const C *);
R   *add_pro(R *, const R *, const R *);
R   *sub_pro(R *, const R *, const R *);

R *force(R *, R, R, R);
C *force(C *, R, R, R);
C *scale(C *, R);
R *scale(R *, R);

void step(R, R, R, R, R);

#endif
