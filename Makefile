# cinfer: C inference engine for concurrent agents on constrained hardware.
#   make            build library + tools (Metal backend on macOS)
#   make NO_METAL=1 CPU-only build (e.g. Raspberry Pi)
CC      ?= clang
UNAME   := $(shell uname -s)
CFLAGS  ?= -O3 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -mcpu=native -D_GNU_SOURCE -D_DARWIN_C_SOURCE
LDLIBS  := -lpthread -lm
BUILD   := build

SRC     := $(wildcard src/*.c)
OBJ     := $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))

ifeq ($(UNAME),Darwin)
ifndef NO_METAL
CFLAGS  += -DCI_METAL=1
OBJ     += $(BUILD)/metal.o
LDLIBS  += -framework Metal -framework Foundation
endif
endif

TOOLS   := $(patsubst tools/%.c,$(BUILD)/%,$(wildcard tools/*.c))

all: $(BUILD)/libcinfer.a $(TOOLS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Metal kernels are embedded as a C string so the binary is self-contained.
$(BUILD)/kernels_metal.h: src/metal/kernels.metal | $(BUILD)
	cd src/metal && xxd -i kernels.metal > ../../$@

$(BUILD)/metal.o: src/metal/metal.m src/*.h $(BUILD)/kernels_metal.h | $(BUILD)
	$(CC) $(CFLAGS) -fobjc-arc -I$(BUILD) -Isrc -c $< -o $@

$(BUILD)/libcinfer.a: $(OBJ)
	ar rcs $@ $^

$(BUILD)/%: tools/%.c $(BUILD)/libcinfer.a
	$(CC) $(CFLAGS) -Isrc $< $(BUILD)/libcinfer.a $(LDLIBS) -o $@

src/unicode_data.c: scripts/gen_unicode.py
	python3 scripts/gen_unicode.py > $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
