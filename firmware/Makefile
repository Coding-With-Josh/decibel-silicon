# firmware - fixed-point noise reduction port (Decibel Silicon)
#
# Targets:
#   gen-tables : regenerate tables.h + golden.h from the validated Python
#                reference (requires ../dsp/.venv).
#   host-test  : build and run the golden comparison on the host (macOS clang).
#   cross      : rv32imc freestanding objects + bare-metal link check.
#   all        : gen-tables + host-test + cross.

PY   := ../dsp/.venv/bin/python
CCX  := riscv64-elf-gcc
XARCH := -march=rv32imc -mabi=ilp32
SRC  := $(wildcard src/*.c)

.PHONY: all gen-tables host-test cross clean

all: gen-tables host-test cross

# --- regenerate generated tables/golden (host; needs numpy) ----------------
gen-tables:
	@mkdir -p build
	$(PY) scripts/gen_tables.py

# --- host golden test (macOS clang, real libc) ------------------------------
host-test: gen-tables
	@mkdir -p build
	cc -O2 -std=c11 -Wall -Wextra -Isrc -Itest \
	   src/fft.c src/spectral_subtract.c test/host_test.c \
	   -o build/host_test -lm
	./build/host_test

# --- rv32imc freestanding cross build + link check --------------------------
cross: gen-tables
	@mkdir -p build
	$(CCX) $(XARCH) -Os -std=c11 -ffreestanding -Wall -Wextra -Isrc -Itest -c \
	    src/fft.c -o build/fft.o
	$(CCX) $(XARCH) -Os -std=c11 -ffreestanding -Wall -Wextra -Isrc -Itest -c \
	    src/spectral_subtract.c -o build/spectral_subtract.o
	$(CCX) $(XARCH) -Os -std=c11 -ffreestanding -Wall -Wextra -Isrc -Itest -c \
	    test/fw_linktest.c -o build/fw_linktest.o
	$(CCX) $(XARCH) -Os -ffreestanding -nostdlib -Wl,-e,_start \
	    build/fft.o build/spectral_subtract.o build/fw_linktest.o \
	    -o build/fw_linktest.elf
	@echo "cross build OK:"
	riscv64-elf-size build/fft.o build/spectral_subtract.o build/fw_linktest.elf

clean:
	rm -rf build