###############################################################################
# Compile setting with intel compiler and intel MPI.
# Modify flags according to the system and environment
###############################################################################

# FC = mpiifort

# FLAG  = -O3 -fPIC -cpp
# FLAG += -g
# FLAG += -fcheck=all -warn-all 
# FLAG += -fp-model=precise 
# FLAG += -check bounds
# FLAG += -xMIC-AVX512

###############################################################################
# Compile setting with nvfortran and non-intel MPI.
# Modify flags according to the system and environment
###############################################################################

FC = mpifort

CPU_ARCH_FLAGS ?= -tp=x86-64-v3
GPU_ARCH_FLAGS ?= -gpu=ccall

FLAG  = -O3 -fPIC -cpp 
FLAG += $(CPU_ARCH_FLAGS)

###############################################################################
# To use CUDA
# PaScaL_TDMA is compiled for CUDA also.
###############################################################################

USE_CUDA = yes
FLAG += -mp:multicore -cuda $(GPU_ARCH_FLAGS) -Minfo=accel 
#FLAG += -DUSE_NVTX

