#ifndef IHD_H
#define IHD_H

#define TWO_PI (6.2831853071795865f)

typedef int Z;
typedef float R;
typedef struct {R r, i;} C;

extern Z M, N1, N2, H2, F2;
extern R *w, *Host;
extern C *W;

void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

void setup(Z, Z);

R *init(R *, R (*)(R, R));
R *load(R *, Z);
Z  dump(Z, R *);

C *scal(C *, R);
R *scal(R *, R);

C *der(C *, C *, Z);

#endif
