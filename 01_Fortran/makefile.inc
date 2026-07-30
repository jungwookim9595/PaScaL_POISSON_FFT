# MPI를 사용하는 주요 컴파일러. Intel MPI의 경우 mpiifort 사용.
FC = mpif90

# 컴파일 플래그

# 디버깅용
#FLAG = -O0 -C -g -traceback -Mpreprocess -Mchkptr -Mchkstk -Mcoff -Kieee -fPIC -Minform=warn -gpu=debug

# 최적화된 실행용 
CPU_ARCH_FLAGS ?= -tp=x86-64-v3
GPU_ARCH_FLAGS ?=
FLAG = -O3 -Munroll -Mpreprocess -fast -fPIC -fortranlibs -Kieee -Mlarge_arrays $(CPU_ARCH_FLAGS)
