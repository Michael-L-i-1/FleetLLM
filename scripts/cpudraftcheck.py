"""Validate CPU DFlash against a dense PyTorch reference and engine rollback.

Run after make NO_METAL=1: python3 scripts/cpudraftcheck.py
Requires numpy/torch only for this development check; inference has no new deps.
"""
import json
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(713)
torch.set_num_threads(1)


def write_st(path, tensors):
    header, chunks, offset = {}, [], 0
    dtype = {"float32": "F32", "float16": "F16", "uint32": "U32"}
    for name, value in tensors.items():
        data = value.tobytes()
        header[name] = dict(dtype=dtype[str(value.dtype)], shape=list(value.shape), data_offsets=[offset, offset + len(data)])
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def matrix(tensors, name, rows, cols):
    codes = rng.integers(0, 16, (rows, cols), dtype=np.uint32)
    scales = np.full((rows, cols // 64), .01, dtype=np.float16)
    biases = np.full_like(scales, -.075)
    packed = np.zeros((rows, cols // 8), dtype=np.uint32)
    for i in range(8):
        packed |= codes[:, i::8] << (4 * i)
    tensors[name + ".weight"] = packed
    tensors[name + ".scales"] = scales
    tensors[name + ".biases"] = biases
    return torch.tensor(codes.astype(np.float32) * np.repeat(scales.astype(np.float32), 64, axis=1)
                        + np.repeat(biases.astype(np.float32), 64, axis=1))


def fixture(path, draft):
    path.mkdir()
    tensors, weights = {}, {}
    cfg = dict(model_type="muse_glimmer_assistant" if draft else "llama", hidden_size=64,
               intermediate_size=128, num_hidden_layers=2 if draft else 3, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5, rope_theta=10000,
               tie_word_embeddings=False)
    prefix = "" if draft else "model."
    if draft:
        cfg.update(target_layer_ids=[0, 2], mask_token_id=63, block_size=4, sliding_window=5,
                   rope_parameters=dict(rope_theta=10000))
        weights["fc"] = matrix(tensors, "encoder.fc", 64, 128)
        tensors["encoder.output_norm_enc.weight"] = np.linspace(.8, 1.2, 64, dtype=np.float32)
    else:
        tensors["model.embed_tokens.weight"] = rng.normal(0, .1, (64, 64)).astype(np.float32)
        tensors["lm_head.weight"] = rng.normal(0, .1, (64, 64)).astype(np.float32)
    tensors[prefix + "norm.weight"] = np.linspace(.9, 1.1, 64, dtype=np.float32)
    for layer in range(cfg["num_hidden_layers"]):
        base = f"{prefix}layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[base + norm + ".weight"] = np.linspace(.8, 1.2, 64, dtype=np.float32)
        for norm in ("q_norm", "k_norm"):
            tensors[base + "self_attn." + norm + ".weight"] = np.linspace(.7, 1.3, 16, dtype=np.float32)
        for name, rows, cols in (("self_attn.q_proj", 64, 64), ("self_attn.k_proj", 32, 64),
                                 ("self_attn.v_proj", 32, 64), ("self_attn.o_proj", 64, 64),
                                 ("mlp.gate_proj", 128, 64), ("mlp.up_proj", 128, 64),
                                 ("mlp.down_proj", 64, 128)):
            weights[base + name] = matrix(tensors, base + name, rows, cols)
    (path / "config.json").write_text(json.dumps(cfg))
    write_st(path / "model.safetensors", tensors)
    return cfg, {k: torch.tensor(v.astype(np.float32)) for k, v in tensors.items() if v.dtype != np.uint32}, weights


def norm(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w


def reference(t, w, target, capture, ctx, blk, elem):
    context = norm(torch.tensor(capture)[ctx[:, 2]] @ w["fc"].T, t["encoder.output_norm_enc.weight"])
    x = target["model.embed_tokens.weight"][blk[:, 2]].clone()
    all_rows = np.concatenate((ctx, blk))
    positions = torch.tensor(all_rows[:, 1], dtype=torch.float32)
    angle = positions[:, None] * (10000 ** (-torch.arange(8, dtype=torch.float32) / 8))[None, :]

    def rope(v, angle):
        a, b = v[..., :8], v[..., 8:]
        cs, sn = angle.cos()[:, None], angle.sin()[:, None]
        return torch.cat((a * cs - b * sn, a * sn + b * cs), -1)

    def stored(v):
        if elem == 4:
            return v
        if elem == 2:
            return v.half().float()
        scale = (v.abs().amax(-1, keepdim=True) / 127).half().float()
        scale[scale == 0] = 1
        return (v / scale).round().clamp(-127, 127) * scale

    for li in range(2):
        base = f"layers.{li}."
        h = norm(x, t[base + "input_layernorm.weight"])
        inputs = torch.cat((context, h))
        k = (inputs @ w[base + "self_attn.k_proj"].T).reshape(-1, 2, 16)
        v = stored((inputs @ w[base + "self_attn.v_proj"].T).reshape(-1, 2, 16))
        k = stored(rope(norm(k, t[base + "self_attn.k_norm.weight"]), angle))
        q = (h @ w[base + "self_attn.q_proj"].T).reshape(-1, 4, 16)
        q = rope(norm(q, t[base + "self_attn.q_norm.weight"]), angle[len(ctx):])
        att = torch.empty_like(q)
        for i, row in enumerate(blk):
            mask = ((all_rows[:, 0] == row[0]) & (all_rows[:, 1] <= row[3]) & (all_rows[:, 1] >= row[3] + 1 - 5))
            for head in range(4):
                scores = k[mask, head // 2] @ q[i, head] / 4
                att[i, head] = scores.softmax(0) @ v[mask, head // 2]
        x = x + att.flatten(1) @ w[base + "self_attn.o_proj"].T
        h = norm(x, t[base + "post_attention_layernorm.weight"])
        gate, up = h @ w[base + "mlp.gate_proj"].T, h @ w[base + "mlp.up_proj"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w[base + "mlp.down_proj"].T
    selected = x[[i for i in range(len(blk)) if i % 4]]
    ids = (norm(selected, t["norm.weight"]) @ target["lm_head.weight"].T).argmax(-1)
    return x.numpy(), ids.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Copy generated fixtures for a native Pi check")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fleet-cpudraft-") as directory:
        root = Path(directory)
        build = root / "build"
        subprocess.run(["make", "NO_METAL=1", f"BUILD={build}", "-j4"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        _, target, _ = fixture(root / "target", False)
        _, tensors, weights = fixture(root / "draft", True)
        ctx = np.array([(seq, p, 0, p) for seq, length in enumerate((17, 19)) for p in range(length)], dtype=np.int32)
        ctx[:, 2] = rng.permutation(len(ctx))
        blk = np.array([(seq, p, 1 if p == length else 63, length + 3)
                        for seq, length in enumerate((17, 19)) for p in range(length, length + 4)], dtype=np.int32)
        capture = rng.normal(0, .3, (len(ctx), 128)).astype(np.float32)
        inputs = root / "inputs.bin"
        inputs.write_bytes(struct.pack("<3i", len(ctx), len(blk), 4) + capture.tobytes() + ctx.tobytes() + blk.tobytes())
        binary = root / "check"
        subprocess.run([os.environ.get("CC", "cc"), "-O1", "-g", "-fsanitize=address,undefined", "-Isrc",
                        "scripts/cpudraftcheck.c", str(build / "libfleetllm.a"), "-lpthread", "-lm", "-o", str(binary)], cwd=ROOT, check=True)
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        for elem in (4, 2, 1):
            expected, ids = reference(tensors, weights, target, capture, ctx, blk, elem)
            for split in (0, 1):
                output = root / "output.bin"
                subprocess.run([str(binary), str(root / "target"), str(root / "draft"), str(inputs), str(output), str(elem), str(split)], check=True, env=env)
                data = output.read_bytes()
                actual = np.frombuffer(data[:len(blk) * 64 * 4], dtype=np.float32).reshape(len(blk), 64)
                actual_ids = np.frombuffer(data[len(blk) * 64 * 4:], dtype=np.int32)
                np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
                np.testing.assert_array_equal(actual_ids, ids)
                print(f"dense PyTorch reference: KV{elem}, split-context={split}, hidden states + tokens PASS", flush=True)
        subprocess.run([str(binary), "engine", str(root / "target"), str(root / "draft")], check=True, env=env)
        if args.export:
            for name in ("target", "draft"):
                shutil.copytree(root / name, args.export / name, dirs_exist_ok=True)
            shutil.copy2(inputs, args.export / inputs.name)


if __name__ == "__main__":
    main()
