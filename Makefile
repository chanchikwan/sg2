CUDA = $(shell if [[ `hostname` =~ platon.* ]]; \
	then echo /sw/pkg/cuda; \
	else echo /usr/local/cuda; fi)
NVCC = $(CUDA)/bin/nvcc
LINK = $(addprefix -Xlinker , -lcufft -rpath $(CUDA)/lib)
FLGS = $(addprefix --compiler-options , -Wall) -O3 \
       $(LINK)$(shell if [ `uname` = Linux ]; then echo 64; fi)

double:
	mkdir -p bin
	$(NVCC) src/*.cu $(FLGS) -o bin/ihd -DOUBLE -arch sm_13

float:
	mkdir -p bin
	$(NVCC) src/*.cu $(FLGS) -o bin/ihd

clean:
	-rm -f bin/ihd
	-rm -f src/*~
	-if [ -z "`ls bin 2>&1`" ]; then rmdir bin; fi
