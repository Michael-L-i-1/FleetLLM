# FleetLLM: high-throughput inference for fleets of agents on one machine.
#   make            build library + `fleet` command (Metal backend on macOS)
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

# Each tools/<name>.c is a subcommand, built as build/fleet-<name> (_ -> -);
# build/fleet (tools/fleet/main.c) dispatches `fleet <name>` to it, git-style.
TOOL_SRC := $(wildcard tools/*.c)
tool_bin = $(BUILD)/fleet-$(subst _,-,$(patsubst tools/%.c,%,$(1)))
TOOLS   := $(foreach t,$(TOOL_SRC),$(call tool_bin,$(t)))

all: $(BUILD)/libfleetllm.a $(TOOLS) $(BUILD)/fleet

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Metal kernels are embedded as C strings so the binary is self-contained
# (vendored MLX headers under src/metal/mlx are inlined).
METAL_SRC := $(wildcard src/metal/*.metal) $(shell find src/metal/mlx -name '*.h')
$(BUILD)/kernels_metal.h: $(METAL_SRC) scripts/embed_metal.py | $(BUILD)
	python3 scripts/embed_metal.py src/metal/kernels.metal kernels_metal > $@
	python3 scripts/embed_metal.py src/metal/kernels_nax.metal kernels_nax_metal >> $@

$(BUILD)/metal.o: src/metal/metal.m src/*.h $(BUILD)/kernels_metal.h | $(BUILD)
	$(CC) $(CFLAGS) -fobjc-arc -I$(BUILD) -Isrc -c $< -o $@

$(BUILD)/libfleetllm.a: $(OBJ)
	ar rcs $@ $^

define TOOL_RULE
$(call tool_bin,$(1)): $(1) tools/common.h $(BUILD)/libfleetllm.a
	$$(CC) $$(CFLAGS) -Isrc $$< $(BUILD)/libfleetllm.a $$(LDLIBS) -o $$@
endef
$(foreach t,$(TOOL_SRC),$(eval $(call TOOL_RULE,$(t))))

$(BUILD)/fleet: tools/fleet/main.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@

src/unicode_data.c: scripts/gen_unicode.py
	python3 scripts/gen_unicode.py > $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
