#!/bin/sh
# CPU-only native conformance checks; never builds Metal or loads model weights.
set -eu
cd "$(dirname "$0")/../../.."
flags='-O1 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -D_GNU_SOURCE -D_DARWIN_C_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer'
make NO_METAL=1 BUILD=build-fleetpi-asan CFLAGS="$flags" build-fleetpi-asan/libfleetllm.a
for name in chatglimmercheck serverglimmercheck swacoalescecheck; do
    cc $flags -Isrc "scripts/$name.c" build-fleetpi-asan/libfleetllm.a -lpthread -lm -o "build-fleetpi-asan/$name"
    ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "build-fleetpi-asan/$name"
done
