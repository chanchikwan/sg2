#ifndef SG2_H
#define SG2_H

#define CFL    0.5
#define DT_MIN 1.0e-8

typedef int Z;
#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
typedef double R;
#else
typedef float R;
#endif
typedef struct {R r, i;} C;

#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
#define K(X) (X)
#else
#define K(X) (X##f)
#endif

#define TWO_PI K(6.2831853071795865)

#define MIN(A, B) ((A) < (B) ? (A) : (B))

/* Ooops... global variables */
extern Z N1, N2, H2, F2;
extern uint3 Bsz, Gsz, Hsz;

extern R *w;
extern C *W, *X, *Y, *Host;

/* Error handling in error.cu */
void error(const char *, ...);

/* FFT wrapper in fft.cu */
void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

/* Miscellaneous in usage.cu and misc.cu, mostly used by main.cu */
void usage(int);
void setprefix(const char *);
int valid(const char *);
int frame(const char *);
const char *name(Z);

/* Random number generator */
Z setseed(const char *);
Z setseed(Z);
Z getseed(void);

/* Setup, helpers, drivers, and I/O */
void setup(Z, Z);
int  solve(R, R, R, R, R, Z, Z);
void setdt(R, R);
R    getdt(R, R, R);
void setrk(const char *);

void step(R, R, R, R, R);
R    flop(void);

C *init(C *, const char *);
C *load(C *, const char *);
C *dump(const char *, C *);

/* Computation kernels */
void lsRKCNn(const Z, const R *, const R *, const R *, R, R, R, R, R);
void reduce (R *, R *, const R *, const R *);

void getu   (C *, C *, const C *);
void jacobi1(C *, C *, const C *);
void jacobi2(C *, C *, const C *);

R *add_pro(R *, const R *, const R *);
R *sub_pro(R *, const R *, const R *);

R *force(R *, R, R, R);
C *force(C *, R, R, R);
C *scale(C *, R);
R *scale(R *, R);

#endif
