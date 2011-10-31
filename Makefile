CUDA = $(subst /bin/nvcc,,$(shell echo `which nvcc`))
NVCC = $(CUDA)/bin/nvcc
AP64 = $(shell if [ `uname` = Linux ]; then echo 64; fi)

LDFLAGS = $(addprefix -Xlinker , -lcufft -rpath $(CUDA)/lib)$(AP64)
CFLAGS  = $(addprefix --compiler-options , -Wall) -O3

double:
	mkdir -p bin
	$(NVCC) src/*.cu -o bin/ihd $(CFLAGS) $(LDFLAGS) -DOUBLE -arch sm_13

float:
	mkdir -p bin
	$(NVCC) src/*.cu -o bin/ihd $(CFLAGS) $(LDFLAGS)

clean:
	-rm -f bin/ihd
	-rm -f src/*~
	-if [ -z "`ls bin 2>&1`" ]; then rmdir bin; fi
