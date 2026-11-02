#!/bin/sh
# Compile a second copy of the portable kernel for comparison with the native one.
set -eu
cd "$(dirname "$0")/.."
make build/libfleetllm.a
set --
for name in cq_kind_name cq_from_qmat cq_free cq_bytes cq_check cq_quantize cq_isa; do
    set -- "$@" "-D$name=scalar_$name"
done
${CC:-cc} -O3 -Isrc -U__ARM_FEATURE_DOTPROD "$@" -Dcq_matmul_rows=scalar_matmul -c src/cpuq.c -o build/cpuqcheck-scalar.o
if [ "$(uname -s)" = Darwin ]; then set -- -framework Metal -framework Foundation; else set --; fi
${CC:-cc} -O3 -Isrc scripts/cpuqcheck.c build/cpuqcheck-scalar.o build/libfleetllm.a -lpthread -lm "$@" -o build/cpuqcheck
build/cpuqcheck
