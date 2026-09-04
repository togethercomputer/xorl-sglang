"""ZORL noise layout v2 ("philox_subseed_v2") — sglang-side twin.

Per-(seed, raw_name) sub-seeded philox4x32-10 standard-normal streams, shared
bit-for-bit with the XoRL parameter-server fold (``src/xorl/server/zorl.py``
and ``zorl_philox_triton.py``). The fold uses exactly the noise a scorer serves; if
either side drifts, its own fixture test fails first. Contract:

  sub_seed = blake2b-64("zorl-noise/philox_subseed_v2:{seed}:{raw_name}")
  value[i] = fp32 BoxMuller(philox4x32_10(key=sub_seed, counter=i//4))[i%4]
  uniforms: u = (u32 + 0.5) / 2**32; TWO_PI = 6.2831855 (fp32)

Run `python -m sglang.srt.lora.zorl_philox` for the self-test.
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Union

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False

ZORL_NOISE_LAYOUT_V1 = "sequential_v1"
ZORL_NOISE_LAYOUT_V2 = "philox_subseed_v2"

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_U32 = 0xFFFFFFFF


def zorl_param_subseed(seed: int, raw_name: str) -> int:
    digest = hashlib.blake2b(
        f"zorl-noise/{ZORL_NOISE_LAYOUT_V2}:{int(seed)}:{raw_name}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _u32_mulhilo(a: torch.Tensor, m: int):
    m_lo = m & 0xFFFF
    m_hi = (m >> 16) & 0xFFFF
    p_lo = a * m_lo
    p_hi = a * m_hi
    lo = (p_lo + ((p_hi & 0xFFFF) << 16)) & _U32
    carry = (p_lo + ((p_hi & 0xFFFF) << 16)) >> 32
    hi = ((p_hi >> 16) + carry) & _U32
    return hi, lo


def _philox4x32_10_batch(idx: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    B = keys.shape[0]
    c0 = (idx & _U32).unsqueeze(0).expand(B, -1).contiguous()
    c1 = ((idx >> 32) & _U32).unsqueeze(0).expand(B, -1).contiguous()
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = (keys & _U32).unsqueeze(1)
    k1 = ((keys >> 32) & _U32).unsqueeze(1)
    w0 = 0
    w1 = 0
    for _ in range(10):
        hi0, lo0 = _u32_mulhilo(c0, _PHILOX_M0)
        hi1, lo1 = _u32_mulhilo(c2, _PHILOX_M1)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ ((k0 + w0) & _U32)) & _U32,
            lo1,
            (hi0 ^ c3 ^ ((k1 + w1) & _U32)) & _U32,
            lo0,
        )
        w0 = (w0 + _PHILOX_W0) & _U32
        w1 = (w1 + _PHILOX_W1) & _U32
    return torch.stack([c0, c1, c2, c3], dim=-1)


if HAS_TRITON:

    @triton.jit
    def _philox_randn_kernel(
        out_ptr,
        keys_ptr,
        numel,
        n_counters,
        counter_offset,
        BLOCK: tl.constexpr,
    ):
        pid_key = tl.program_id(0)
        pid_blk = tl.program_id(1)
        key = tl.load(keys_ptr + pid_key)
        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)

        cpos = pid_blk * BLOCK + tl.arange(0, BLOCK)
        cmask = cpos < n_counters
        cidx = (counter_offset + cpos).to(tl.int64)
        c0 = (cidx & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((cidx >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c2 = tl.zeros([BLOCK], dtype=tl.uint32)
        c3 = tl.zeros([BLOCK], dtype=tl.uint32)

        M0: tl.constexpr = 0xD2511F53
        M1: tl.constexpr = 0xCD9E8D57
        W0: tl.constexpr = 0x9E3779B9
        W1: tl.constexpr = 0xBB67AE85
        for _ in tl.static_range(10):
            hi0 = tl.umulhi(c0, M0)
            lo0 = c0 * M0
            hi1 = tl.umulhi(c2, M1)
            lo1 = c2 * M1
            nc0 = hi1 ^ c1 ^ k0
            nc2 = hi0 ^ c3 ^ k1
            c0, c1, c2, c3 = nc0, lo1, nc2, lo0
            k0 = k0 + W0
            k1 = k1 + W1

        inv = 2.3283064365386963e-10  # 1 / 2**32
        u0 = (c0.to(tl.float32) + 0.5) * inv
        u1 = (c1.to(tl.float32) + 0.5) * inv
        u2 = (c2.to(tl.float32) + 0.5) * inv
        u3 = (c3.to(tl.float32) + 0.5) * inv
        TWO_PI: tl.constexpr = 6.2831855
        r0 = tl.sqrt(-2.0 * tl.log(u0))
        t0 = TWO_PI * u1
        r1 = tl.sqrt(-2.0 * tl.log(u2))
        t1 = TWO_PI * u3
        z0 = r0 * tl.cos(t0)
        z1 = r0 * tl.sin(t0)
        z2 = r1 * tl.cos(t1)
        z3 = r1 * tl.sin(t1)

        base = pid_key.to(tl.int64) * numel + cpos.to(tl.int64) * 4
        tl.store(out_ptr + base + 0, z0, mask=cmask & (cpos * 4 + 0 < numel))
        tl.store(out_ptr + base + 1, z1, mask=cmask & (cpos * 4 + 1 < numel))
        tl.store(out_ptr + base + 2, z2, mask=cmask & (cpos * 4 + 2 < numel))
        tl.store(out_ptr + base + 3, z3, mask=cmask & (cpos * 4 + 3 < numel))



def zorl_philox_randn_batch(
    sub_seeds: List[int],
    numel: int,
    *,
    device: Union[str, torch.device] = "cpu",
    counter_offset: int = 0,
) -> torch.Tensor:
    """[B, numel] deterministic normals; Triton kernel on CUDA, torch fallback."""
    B = len(sub_seeds)
    if numel <= 0 or B == 0:
        return torch.empty(B, max(numel, 0), dtype=torch.float32, device=device)
    if torch.device(device).type == "cuda" and HAS_TRITON:
        out = torch.empty(B, numel, dtype=torch.float32, device=device)
        keys = torch.tensor(sub_seeds, device=device, dtype=torch.int64)
        n_counters = (numel + 3) // 4
        BLOCK = 1024
        grid = (B, triton.cdiv(n_counters, BLOCK))
        _philox_randn_kernel[grid](out, keys, numel, n_counters, counter_offset, BLOCK=BLOCK)
        return out
    keys = torch.tensor(sub_seeds, device=device, dtype=torch.int64)
    n_counters = (numel + 3) // 4
    outs = []
    SLAB = 1 << 22
    out = torch.empty(B, n_counters * 4, dtype=torch.float32, device=device)
    two_pi = 6.2831855
    for slab_start in range(0, n_counters, SLAB):
        slab_n = min(SLAB, n_counters - slab_start)
        idx = torch.arange(
            counter_offset + slab_start, counter_offset + slab_start + slab_n,
            device=device, dtype=torch.int64,
        )
        u32 = _philox4x32_10_batch(idx, keys)
        u = (u32.to(torch.float32) + 0.5) * (1.0 / 4294967296.0)
        r0 = torch.sqrt(-2.0 * torch.log(u[..., 0]))
        t0 = two_pi * u[..., 1]
        r1 = torch.sqrt(-2.0 * torch.log(u[..., 2]))
        t1 = two_pi * u[..., 3]
        z = torch.stack(
            [r0 * torch.cos(t0), r0 * torch.sin(t0), r1 * torch.cos(t1), r1 * torch.sin(t1)],
            dim=-1,
        )
        out[:, slab_start * 4 : (slab_start + slab_n) * 4] = z.reshape(B, -1)
    return out[:, :numel]


# Cross-repo fixture (torch CPU reference values; the PS pins the SAME):
_FIXTURE_SEED = 1234567
_FIXTURE_RAW = "model.layers.0.mlp.experts.gate_up_proj.lora_A"
_FIXTURE_FIRST4 = [-2.23556876, -0.98627412, 0.45100215, 0.94255328]


def _self_test() -> None:
    key = zorl_param_subseed(_FIXTURE_SEED, _FIXTURE_RAW)
    z = zorl_philox_randn_batch([key], 256)[0]
    expected = torch.tensor(_FIXTURE_FIRST4)
    assert torch.allclose(z[:4], expected, atol=1e-6), z[:4].tolist()
    mid = zorl_philox_randn_batch([key], 64, counter_offset=16)[0]
    assert torch.equal(mid, z[64:128])
    print("zorl_philox self-test OK; fixture", [round(v, 8) for v in z[:4].tolist()])


if __name__ == "__main__":
    _self_test()


if HAS_TRITON:

    @triton.jit
    def _philox_randn_segments_kernel(
        out_ptr,
        blk_key_ptr,      # [n_blocks] int64: philox key for this block's segment
        blk_out_ptr,      # [n_blocks] int64: output offset of this block's first value
        blk_cstart_ptr,   # [n_blocks] int64: this block's first counter (segment-relative)
        blk_remain_ptr,   # [n_blocks] int64: values remaining in segment from this block on
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        key = tl.load(blk_key_ptr + pid)
        out_base = tl.load(blk_out_ptr + pid)
        c_start = tl.load(blk_cstart_ptr + pid)
        remain = tl.load(blk_remain_ptr + pid)

        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)
        cpos = tl.arange(0, BLOCK)
        cidx = (c_start + cpos).to(tl.int64)
        c0 = (cidx & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((cidx >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c2 = tl.zeros([BLOCK], dtype=tl.uint32)
        c3 = tl.zeros([BLOCK], dtype=tl.uint32)

        M0: tl.constexpr = 0xD2511F53
        M1: tl.constexpr = 0xCD9E8D57
        W0: tl.constexpr = 0x9E3779B9
        W1: tl.constexpr = 0xBB67AE85
        for _ in tl.static_range(10):
            hi0 = tl.umulhi(c0, M0)
            lo0 = c0 * M0
            hi1 = tl.umulhi(c2, M1)
            lo1 = c2 * M1
            nc0 = hi1 ^ c1 ^ k0
            nc2 = hi0 ^ c3 ^ k1
            c0, c1, c2, c3 = nc0, lo1, nc2, lo0
            k0 = k0 + W0
            k1 = k1 + W1

        inv = 2.3283064365386963e-10
        u0 = (c0.to(tl.float32) + 0.5) * inv
        u1 = (c1.to(tl.float32) + 0.5) * inv
        u2 = (c2.to(tl.float32) + 0.5) * inv
        u3 = (c3.to(tl.float32) + 0.5) * inv
        TWO_PI: tl.constexpr = 6.2831855
        r0 = tl.sqrt(-2.0 * tl.log(u0))
        t0 = TWO_PI * u1
        r1 = tl.sqrt(-2.0 * tl.log(u2))
        t1 = TWO_PI * u3
        z0 = r0 * tl.cos(t0)
        z1 = r0 * tl.sin(t0)
        z2 = r1 * tl.cos(t1)
        z3 = r1 * tl.sin(t1)

        vpos = cpos.to(tl.int64) * 4
        tl.store(out_ptr + out_base + vpos + 0, z0, mask=vpos + 0 < remain)
        tl.store(out_ptr + out_base + vpos + 1, z1, mask=vpos + 1 < remain)
        tl.store(out_ptr + out_base + vpos + 2, z2, mask=vpos + 2 < remain)
        tl.store(out_ptr + out_base + vpos + 3, z3, mask=vpos + 3 < remain)


class PhiloxSegmentGeometry:
    """Seed-independent launch geometry for a fixed list of segment sizes.

    Built once per (adapter, marker) and cached by the caller; per draw only
    the per-segment KEYS change (sub_seed depends on the seed), so a draw is
    one small H2D (keys gather) + ONE kernel launch regardless of how many
    raw entries the adapter has.
    """

    BLOCK = 1024

    def __init__(self, numels: List[int], device):
        self.numels = [int(n) for n in numels]
        self.device = torch.device(device)
        self.total = sum(self.numels)
        self.out_offsets = []
        blk_seg, blk_out, blk_cstart, blk_remain = [], [], [], []
        out = 0
        for seg_idx, n in enumerate(self.numels):
            self.out_offsets.append(out)
            n_counters = (n + 3) // 4
            n_blocks = (n_counters + self.BLOCK - 1) // self.BLOCK
            for b in range(n_blocks):
                blk_seg.append(seg_idx)
                blk_cstart.append(b * self.BLOCK)
                blk_out.append(out + b * self.BLOCK * 4)
                blk_remain.append(n - b * self.BLOCK * 4)
            out += n
        self.n_blocks = len(blk_seg)
        self.blk_seg = torch.tensor(blk_seg, device=self.device, dtype=torch.int64)
        self.blk_out = torch.tensor(blk_out, device=self.device, dtype=torch.int64)
        self.blk_cstart = torch.tensor(blk_cstart, device=self.device, dtype=torch.int64)
        self.blk_remain = torch.tensor(blk_remain, device=self.device, dtype=torch.int64)

    def draw(self, keys: List[int]) -> torch.Tensor:
        """One flat fp32 tensor holding every segment's normals, one launch."""
        if len(keys) != len(self.numels):
            raise ValueError(f"expected {len(self.numels)} keys, got {len(keys)}")
        if not (HAS_TRITON and self.device.type == "cuda"):
            # Reference/CPU fallback: per-segment batch draws (bit-identical).
            out = torch.empty(self.total, dtype=torch.float32, device=self.device)
            for key, n, off in zip(keys, self.numels, self.out_offsets):
                out[off : off + n] = zorl_philox_randn_batch([key], n, device=self.device)[0]
            return out
        seg_keys = torch.tensor(keys, device=self.device, dtype=torch.int64)
        blk_key = seg_keys[self.blk_seg]
        out = torch.empty(self.total, dtype=torch.float32, device=self.device)
        _philox_randn_segments_kernel[(self.n_blocks,)](
            out, blk_key, self.blk_out, self.blk_cstart, self.blk_remain, BLOCK=self.BLOCK
        )
        return out
