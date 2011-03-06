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

extern Z N1, N2, H2, F2, Seed;
extern uint3 Bsz, Gsz, Hsz;

extern R *w, *Host;
extern C *W, *X, *Y;

extern char Prefix[];

void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

const char *name(Z);
int  exist(const char *);

void setup(Z, Z);
R    getdt(R, R, R);
void step(R, R, R, R, R);
R   *init(R *, R (*)(R, R));
R   *load(R *, const char *);
R   *dump(const char *, R *);

R *force(R *, R, R, R);
C *force(C *, R, R, R);
C *scale(C *, R);
R *scale(R *, R);
C *deriv(C *, C *, Z);

void dx_dd_dy(C *, C *, C *);
void dy_dd_dx(C *, C *, C *);
R    *sub_pro(R *, R *, R *);
R    *add_pro(R *, R *, R *);

#endif
