# Static Poisson-operator TDMA package.  The recommended target builds the
# direct-C2I debug and performance executables together.  Dynamic/static TDMA
# A/B selection is runtime-controlled by POISSON_TDMA_STATIC_OPERATOR.
#
#   make static-tdma
#     run/examples_debug_c2i_direct.exe
#     run/examples_perf_c2i_direct.exe
#
# The older four-way C2I target is retained as an optional source-level A/B.

include ./makefile.inc

DEBUG_FLAGS = -DPOISSON_DETAILED_PROFILE -DDEBUG_BUILD
PERF_FLAGS  = -DPERF_BUILD -DPOISSON_COARSE_PROFILE
DIRECT_FLAG = -DPOISSON_DIRECT_C2I_FFT

all: static-tdma

static-tdma: debug-c2i-direct perf-c2i-direct

c2i-ab: debug-c2i-legacy debug-c2i-direct \
	perf-c2i-legacy perf-c2i-direct

debug-c2i-legacy:
	mkdir -p include/debug_c2i_legacy lib/debug_c2i_legacy run
	$(MAKE) -C PaScaL_TDMA lib MODE=debug_c2i_legacy BUILD_FLAGS="$(DEBUG_FLAGS)"
	$(MAKE) -C src all MODE=debug_c2i_legacy BUILD_FLAGS="$(DEBUG_FLAGS)"
	$(MAKE) -C examples all MODE=debug_c2i_legacy BUILD_FLAGS="$(DEBUG_FLAGS)"

debug-c2i-direct:
	mkdir -p include/debug_c2i_direct lib/debug_c2i_direct run
	$(MAKE) -C PaScaL_TDMA lib MODE=debug_c2i_direct BUILD_FLAGS="$(DEBUG_FLAGS) $(DIRECT_FLAG)"
	$(MAKE) -C src all MODE=debug_c2i_direct BUILD_FLAGS="$(DEBUG_FLAGS) $(DIRECT_FLAG)"
	$(MAKE) -C examples all MODE=debug_c2i_direct BUILD_FLAGS="$(DEBUG_FLAGS) $(DIRECT_FLAG)"

perf-c2i-legacy:
	mkdir -p include/perf_c2i_legacy lib/perf_c2i_legacy run
	$(MAKE) -C PaScaL_TDMA lib MODE=perf_c2i_legacy BUILD_FLAGS="$(PERF_FLAGS)"
	$(MAKE) -C src all MODE=perf_c2i_legacy BUILD_FLAGS="$(PERF_FLAGS)"
	$(MAKE) -C examples all MODE=perf_c2i_legacy BUILD_FLAGS="$(PERF_FLAGS)"

perf-c2i-direct:
	mkdir -p include/perf_c2i_direct lib/perf_c2i_direct run
	$(MAKE) -C PaScaL_TDMA lib MODE=perf_c2i_direct BUILD_FLAGS="$(PERF_FLAGS) $(DIRECT_FLAG)"
	$(MAKE) -C src all MODE=perf_c2i_direct BUILD_FLAGS="$(PERF_FLAGS) $(DIRECT_FLAG)"
	$(MAKE) -C examples all MODE=perf_c2i_direct BUILD_FLAGS="$(PERF_FLAGS) $(DIRECT_FLAG)"

# Familiar aliases retained for manual use.
dual: c2i-ab
debug: debug-c2i-direct
perf: perf-c2i-direct
example: c2i-ab

clean:
	$(MAKE) -C PaScaL_TDMA clean
	$(MAKE) -C src clean
	$(MAKE) -C examples clean
	rm -rf include/debug* include/perf*
	rm -rf lib/debug* lib/perf*
	rm -f run/examples_debug_c2i_legacy.exe
	rm -f run/examples_debug_c2i_direct.exe
	rm -f run/examples_perf_c2i_legacy.exe
	rm -f run/examples_perf_c2i_direct.exe
	rm -f run/examples.exe
	rm -f include/*.mod lib/*.a
	rm -f PaScaL_TDMA/include/*.mod PaScaL_TDMA/lib/*.a
	rm -f PaScaL_TDMA/src/libpascal_tdma.a
	rm -f examples/obj/*.o examples/obj/*.mod
	rm -f src/obj/*.o src/obj/*.mod
	rm -f PaScaL_TDMA/src/obj/*.o PaScaL_TDMA/src/obj/*.mod
	rm -rf core.* run/core.*

rm:
	$(MAKE) -C src rm
	$(MAKE) -C examples rm
	rm -rf core.* run/core.*

.PHONY: all static-tdma c2i-ab dual debug perf example \
	debug-c2i-legacy debug-c2i-direct \
	perf-c2i-legacy perf-c2i-direct clean rm
