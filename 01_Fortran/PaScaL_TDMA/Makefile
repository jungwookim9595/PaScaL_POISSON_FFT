# Compiler and compilation flags are included in Makefile.inc.
include Makefile.inc

MODE ?= debug
BUILD_FLAGS ?=

lib:
	mkdir -p include/$(MODE) lib/$(MODE) src/obj/$(MODE)
	$(MAKE) -C src all MODE=$(MODE) BUILD_FLAGS="$(BUILD_FLAGS)"

all: lib

clean:
	$(MAKE) -C src clean
	rm -rf include/debug* include/perf* lib/debug* lib/perf*

.PHONY: lib all clean
