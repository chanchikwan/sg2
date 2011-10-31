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
   along with sg2.  If not, see <http://www.gnu.org/licenses/>. */

#include <stdio.h>
#include <string.h>
#include "sg2.h"

#define SIZE 256

static char prefix[SIZE] = "";

void setprefix(const char *p)
{
  strncat(prefix, p, SIZE);
  prefix[SIZE - 1] = '\0';
}

int valid(const char *name)
{
  const Z k  = (MIN(N1, N2) - 1) / 3;

  FILE *file = fopen(name, "r");
  if(file) {
    Z size[3];
    fread(size, sizeof(Z), 3, file);

  if(size[0] == -(Z)sizeof(C) &&
     size[1] <= 1 + k * 2     &&
     size[2] <= 1 + k         )
      return 1;
  }
  return 0;
}

int frame(const char *h)
{
  char c;
  while((int)(c = *h++))
    if('0' <= c && c <= '9')
      return atoi(h-1); /* get the frame number */
  return 0;
}

const char *name(Z i, const char *ext)
{
  static char n[SIZE];
  snprintf(n, SIZE, "%s%04d.%s", prefix, i, ext);
  return n;
}
