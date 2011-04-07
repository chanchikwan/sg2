CUDA = $(shell if [[ `hostname` =~ platon.* ]]; \
	then echo /sw/pkg/cuda; \
	else echo /usr/local/cuda; fi)
NVCC = $(CUDA)/bin/nvcc
LINK = $(addprefix -Xlinker , -lcufft -rpath $(CUDA)/lib)
FLGS = $(addprefix --compiler-options , -Wall) -O3 \
       $(LINK)$(shell if [ `uname` = Linux ]; then echo 64; fi)

double:
	$(NVCC) *.cu $(FLGS) -o ihd -DOUBLE -arch sm_13

float:
	$(NVCC) *.cu $(FLGS) -o ihd

clean:
	-rm -f ihd
	-rm -f *~
