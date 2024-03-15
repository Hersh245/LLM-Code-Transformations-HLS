CXX=g++
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++17 -O3 -DFASTSIM $(LDFLAGS)

MCC=merlincc
CMP_OPT=-d11 --attribute burst_total_size_threshold=36700160 --attribute burst_single_size_threshold=36700160 -funsafe-math-optimizations
LNK_OPT=-d11
CXX_INC_DIRS=-I ./ -I $(MACH_COMMON_DIR)
KERNEL_INC_DIR=$(CXX_INC_DIRS)  -I $(XILINX_HLS)/lnx64/tools/clang-3.9/lib/gcc/x86_64-unknown-linux-gnu/4.8.2/include/ -I $(XILINX_HLS)/include/  -I /opt/merlin/sources/merlin-compiler/trunk/source-opt/include/apint_include/

VENDOR=XILINX
DEVICE=xilinx_u200_gen3x16_xdma_2_202110_1
ifeq ($(wildcard $(AWS_PLATFORM)),)
    DEVICE=xilinx_u200_gen3x16_xdma_2_202110_1
	PATH_D=/opt/xilinx/platforms/$(DEVICE)/$(DEVICE).xpfm
else
    PATH_D=$(AWS_PLATFORM)
endif

KERNEL ?= cnn
ifeq ($(KERNEL), cnn)
	SRCS=lib/cnn.h lib/cnn.cpp lib/main.cpp lib/cnn-krnl.h cnn-krnl.cpp
	KERNEL_FILE=cnn-krnl.cpp
else
	SRCS=lib/$(KERNEL)-krnl.h lib/$(KERNEL)-krnl.cpp lib/$(KERNEL)-main.cpp
	KERNEL_FILE=lib/$(KERNEL)-krnl.cpp
endif

test: $(KERNEL)
	./$<

$(KERNEL): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

estimate: merlin.rpt
	grep -m 1 -B 1 -A 3 "Cycles" merlin.rpt

merlin.rpt: $(KERNEL)-krnl.mco
	$(MCC) $^ --report=estimate $(LNK_OPT) -p=$(PATH_D) --kernel_frequency 250

$(KERNEL)-krnl.mco: $(KERNEL_FILE)
	$(MCC) -c $^ -D $(VENDOR) -o $(KERNEL)-krnl $(CMP_OPT) -p=$(PATH_D) $(KERNEL_INC_DIR) 

clean:
	$(RM) merlin.rpt merlin.log
	$(RM) cnn vadd dotprod
	$(RM) __merlin*.h *.so *.mco
	$(RM) xilinx_com_hls_*.zip
	$(RM) -r .merlin_prj .Mer