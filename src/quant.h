// Reference (scalar) dequantization for GGUF tensor types.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

float fp16_to_f32(uint16_t h);
uint16_t f32_to_fp16(float f);

// Dequantize n elements (a multiple of the type's block size) of a row.
bool dequant_row(uint32_t type, const void *src, float *dst, size_t n);
// Whether dequant_row supports the type.
bool dequant_supported(uint32_t type);
