PROJECT_ROOT = ..
include $(PROJECT_ROOT)/makefile.inc

MODE ?= debug
BUILD_FLAGS ?=

EXE = examples_$(MODE).exe
OBJDIR = ./obj/$(MODE)

OBJS = global mpi_topology interface mpi_subdomain cuda_subdomain \
       para_range cuda_pressure cuda_post main
SRC = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(OBJS)))

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC = -I$(PaScaL_HOME)/include/$(MODE)
PaScaLLIB = -L$(PaScaL_HOME)/lib/$(MODE) -lpascal_tdma

FFT_POI_HOME = ..
FFT_POIINC = -I$(FFT_POI_HOME)/include/$(MODE)
FFT_POILIB = -L$(FFT_POI_HOME)/lib/$(MODE) -lfft_poisson

NVTXFLAGS = -L/usr/local/cuda/lib64
CUDA_FLAGS = -cuda -cudalib=cufft,cublas -Minfo=accel $(GPU_ARCH_FLAGS)
ACC_FLAGS = -acc=gpu
PRECISION_FLAGS = -DDOUBLE_PRECISION
CUDECOMP_HOME ?= ../_install/cudecomp
CUDECOMP_INC = -I$(CUDECOMP_HOME)/include
CUDECOMP_LIB = -L$(CUDECOMP_HOME)/lib \
	       -Wl,-rpath,$(abspath $(CUDECOMP_HOME)/lib) \
	       -lcudecomp_fort -lcudecomp
COMPILE_FLAGS = $(NVTXFLAGS) $(CUDA_FLAGS) $(ACC_FLAGS) \
                $(PRECISION_FLAGS) $(BUILD_FLAGS) \
                $(FFT_POIINC) $(PaScaLINC) $(CUDECOMP_INC)

all: $(OBJDIR) $(SRC)
	$(FC) $(FLAG) -o $(EXE) $(SRC) $(NVTXFLAGS) \
	      $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) \
	      $(FFT_POILIB) $(PaScaLLIB) $(CUDECOMP_LIB) \
	      -lcufft -lcublas
	mv $(EXE) ../run/

$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.o: %.f90 | $(OBJDIR)
	$(FC) $(FLAG) -c $< -o $@ -module $(OBJDIR) -I$(OBJDIR) \
	      $(COMPILE_FLAGS)

# Fortran module dependencies.  Listing source files in OBJS does not impose
# compilation order when make is run with -j.
GLOBAL_OBJ         = $(OBJDIR)/global.o
MPI_TOPOLOGY_OBJ   = $(OBJDIR)/mpi_topology.o
INTERFACE_OBJ      = $(OBJDIR)/interface.o
MPI_SUBDOMAIN_OBJ  = $(OBJDIR)/mpi_subdomain.o
CUDA_SUBDOMAIN_OBJ = $(OBJDIR)/cuda_subdomain.o
CUDA_PRESSURE_OBJ  = $(OBJDIR)/cuda_pressure.o
CUDA_POST_OBJ      = $(OBJDIR)/cuda_post.o
MAIN_OBJ           = $(OBJDIR)/main.o

$(MPI_TOPOLOGY_OBJ): $(GLOBAL_OBJ)
$(INTERFACE_OBJ): $(GLOBAL_OBJ)
$(MPI_SUBDOMAIN_OBJ): $(GLOBAL_OBJ) $(MPI_TOPOLOGY_OBJ) $(INTERFACE_OBJ)
$(CUDA_SUBDOMAIN_OBJ): $(GLOBAL_OBJ) $(MPI_TOPOLOGY_OBJ) \
	$(INTERFACE_OBJ) $(MPI_SUBDOMAIN_OBJ)
$(CUDA_PRESSURE_OBJ): $(GLOBAL_OBJ) $(MPI_SUBDOMAIN_OBJ) \
	$(CUDA_SUBDOMAIN_OBJ)
$(CUDA_POST_OBJ): $(GLOBAL_OBJ) $(MPI_TOPOLOGY_OBJ) \
	$(MPI_SUBDOMAIN_OBJ) $(CUDA_SUBDOMAIN_OBJ)
$(MAIN_OBJ): $(GLOBAL_OBJ) $(MPI_SUBDOMAIN_OBJ) $(CUDA_SUBDOMAIN_OBJ) \
	$(CUDA_PRESSURE_OBJ) $(CUDA_POST_OBJ)

clean:
	rm -rf ./obj/debug* ./obj/perf*
	rm -f $(PROJECT_ROOT)/run/examples_debug_c2i_legacy.exe
	rm -f $(PROJECT_ROOT)/run/examples_debug_c2i_direct.exe
	rm -f $(PROJECT_ROOT)/run/examples_perf_c2i_legacy.exe
	rm -f $(PROJECT_ROOT)/run/examples_perf_c2i_direct.exe
	rm -rf $(PROJECT_ROOT)/run/data

rm:
	rm -rf $(PROJECT_ROOT)/run/data

.PHONY: all clean rm
