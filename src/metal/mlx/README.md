# Vendored MLX Metal headers

Unmodified copies of headers from [MLX](https://github.com/ml-explore/mlx)
v0.32.2 (`mlx/backend/metal/kernels/`), MIT licensed (see `LICENSE`,
Copyright © 2023 Apple Inc.):

- `steel/defines.h`
- `steel/utils/type_traits.h`, `steel/utils/integral_constant.h`
- `steel/gemm/nax.h` — NAX (M5 neural accelerator) tiles over Metal 4
  `mpp::tensor_ops::matmul2d`

They are inlined into the embedded kernel source at build time by
`scripts/embed_metal.py` (includes of `mlx/backend/metal/kernels/...` resolve
here). `src/metal/kernels_nax.metal` adapts MLX's `qmm_t_nax_tgp_impl` and
`QuantizedBlockLoader` (quantized_nax.h) on top of them.
