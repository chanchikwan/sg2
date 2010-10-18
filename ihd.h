#ifndef IHD_H
#define IHD_H

typedef int Z;
typedef float R;
typedef struct {R r, i;} C;

void mkplans(Z, Z);
void rmplans(void);
C   *forward(C *, R *);
R   *inverse(R *, C *);

#endif
