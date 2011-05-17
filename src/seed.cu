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
#include "sg2.h"

static Z rseed, byhand = 0;

Z setseed(const char *s)
{
  FILE *f = fopen(s, "r");
  if(!f) {
    char c;
    while((int)(c = *s++))
      if(c < '0' || '9' < c)
        return -1;
    byhand = 1;
    srand(rseed = atoi(s));
    return rseed;
  }
  fclose(f);
  return -1;
}

Z setseed(Z s)
{
  if(!byhand)
    srand(rseed = s);
  return rseed;
}

Z getseed(void)
{
  srand(rseed = rand());
  return rseed;
}
