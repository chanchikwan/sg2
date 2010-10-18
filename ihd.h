#ifndef IHD_H
#define IHD_H

typedef int Z;
typedef float R;
typedef struct {R r, i;} C;

extern Z N1, N2, H2, F2;
extern R *w, *Host;
extern C *W;

void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

void setup(Z, Z);

#endif
