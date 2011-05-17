#include <stdio.h>
#include <stdarg.h>
#include "sg2.h"

void error(const char *format, ...)
{
  va_list args;

  fflush(stdout);

  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);

  exit(-1);
}
