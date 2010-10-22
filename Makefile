CUDA = /usr/local/cuda
NVCC = $(CUDA)/bin/nvcc
LINK = $(addprefix -Xlinker , -lcufft -rpath $(CUDA)/lib)
FLGS = $(addprefix --compiler-options , -Wall)

double:
	$(NVCC) *.cu -O3 $(LINK) $(FLGS) -o ihd -DOUBLE

float:
	$(NVCC) *.cu -O3 $(LINK) $(FLGS) -o ihd

clean:
	-rm -f ihd
	-rm -f *~
