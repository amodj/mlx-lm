# Copyright © 2023-2024 Apple Inc.
# libKVC Stage A' copy-on-park cache — vanilla mirrors during decode; KV is
# copied into libKVC only at flush/park boundaries (no per-token shadow).
# Supports radix-reused prefix resume: if allocate matches a committed prefix,
# mirrors are restored from libKVC so the caller only prefills the remainder.

import time

import mlx.core as mx
import numpy as np
from .cache import KVCache, _BaseCache


class KvcLayerCache(_BaseCache):
    """Per-layer cache handed to the model; a plain vanilla KVCache mirror.

    The mirror is the source of truth while attached. libKVC receives bytes
    only when the coordinator flushes (park/commit boundaries)."""

    def __init__(self, coord, layer_idx):
        self._coord = coord
        self._idx = layer_idx
        self._mirror = KVCache()

    def update_and_fetch(self, keys, values):
        return self._mirror.update_and_fetch(keys, values)

    @property
    def offset(self):
        return self._mirror.offset

    def make_mask(self, *args, **kwargs):
        return self._mirror.make_mask(*args, **kwargs)

    def size(self):
        return self._mirror.size()

    @property
    def state(self):
        return self._mirror.state

    def is_trimmable(self):
        return False  # libKVC has no trim; never advertise

    def empty(self):
        return self._mirror.empty()

    @property
    def nbytes(self):
        return self._mirror.nbytes


class KvcPromptCache:
    """Coordinator: owns one libKVC sequence; copies mirror KV into libKVC at
    flush/park boundaries (copy-on-park), never per token.

    If allocate matched a committed radix prefix, mirrors are restored from
    libKVC so the caller only needs to prefill tokens[reused_tokens:]."""

    def __init__(self, model, manager, prompt_tokens=None,
                 n_kv_heads=None, head_dim=None):
        self.n_layers = len(model.layers)
        self.manager = manager
        self.seq = manager.allocate(list(prompt_tokens or []))
        self.bpt = manager.kv_bytes_per_token

        if n_kv_heads is not None and head_dim is not None:
            self._h = int(n_kv_heads)
            self._d = int(head_dim)
        else:
            self._h = None
            self._d = None

        self.detached = False
        self.last_park_s = 0.0
        self.caches = [KvcLayerCache(self, i) for i in range(self.n_layers)]

        reused = int(manager.seq_len(self.seq))
        if reused > 0:
            assert reused % manager.block_size == 0, (
                f"radix reuse gave {reused} tokens, not a multiple of "
                f"block_size {manager.block_size}"
            )
            self.offset = reused
            self._restore_mirrors_from_kvc()
        else:
            self.offset = 0

        self.reused_tokens = reused

    # ------------------------------------------------------------------
    # Mirror restore (shared by __init__ radix-resume and attach)
    # ------------------------------------------------------------------

    def _restore_mirrors_from_kvc(self, timeout_ms=30000):
        """Read self.offset tokens from libKVC and populate each layer mirror."""
        T = self.offset
        assert T > 0, "_restore_mirrors_from_kvc called with offset 0"
        assert self._d is not None, (
            "head_dim unknown; pass n_kv_heads/head_dim to make_kvc_prompt_cache"
        )
        raw = self.manager.read(self.seq, timeout_ms=timeout_ms)
        assert len(raw) == T * self.bpt, (
            f"read length {len(raw)} != offset*bpt {T * self.bpt}"
        )
        D = self._d
        L = self.n_layers
        blob = mx.array(
            np.frombuffer(raw, dtype=np.float16).reshape(T, L, 2, -1, D)
        )
        for l, lc in enumerate(self.caches):
            k = mx.contiguous(mx.transpose(blob[:, l, 0], (1, 0, 2)))[None]
            v = mx.contiguous(mx.transpose(blob[:, l, 1], (1, 0, 2)))[None]
            lc._mirror.state = (k, v)
        mx.eval(*[c._mirror.keys for c in self.caches])

    # ------------------------------------------------------------------
    # Copy-on-park
    # ------------------------------------------------------------------

    def _mirror_offset(self):
        offs = {int(lc._mirror.offset) for lc in self.caches}
        assert len(offs) == 1, f"mirror offsets diverged: {sorted(offs)}"
        return offs.pop()

    def flush_to_kvc(self, chunk_tokens=2048):
        """Copy mirror KV for tokens [self.offset, mirror_offset) into libKVC.

        Chunked to bound transient memory. Returns the wall time spent.
        No-op when libKVC is already up to date or the mirrors are empty.
        """
        assert not self.detached, "cannot flush while detached (mirrors are gone)"
        t_start = time.perf_counter()
        total = self._mirror_offset()
        if total <= self.offset:
            self.last_park_s = 0.0
            return 0.0
        mirrors = [lc._mirror for lc in self.caches]
        if self._h is None:
            k0 = mirrors[0].keys
            self._h, self._d = int(k0.shape[1]), int(k0.shape[3])
        for c0 in range(self.offset, total, int(chunk_tokens)):
            c1 = min(c0 + int(chunk_tokens), total)
            ks = mx.stack([m.keys[0, :, c0:c1, :] for m in mirrors])  # (L,H,t,D)
            vs = mx.stack([m.values[0, :, c0:c1, :] for m in mirrors])
            if ks.dtype != mx.float16:
                ks = ks.astype(mx.float16)
                vs = vs.astype(mx.float16)
            k = mx.transpose(ks, (2, 0, 1, 3))  # (t,L,H,D)
            v = mx.transpose(vs, (2, 0, 1, 3))
            rec = mx.contiguous(mx.stack([k, v], axis=2))  # (t,L,2,H,D)
            mx.eval(rec)
            host = np.array(rec, dtype=np.float16, copy=False)
            if not host.flags["C_CONTIGUOUS"] or not host.flags["WRITEABLE"]:
                host = np.ascontiguousarray(host)
                if not host.flags["WRITEABLE"]:
                    host = host.copy()
            assert host.nbytes == (c1 - c0) * self.bpt
            self.manager.append_many(self.seq, host, c0)
            self.offset = c1
        self.last_park_s = time.perf_counter() - t_start
        return self.last_park_s

    def commit_prompt(self, chunk_tokens=2048):
        """Flush then publish the prompt prefix for radix sharing.

        Call once right after prefill. Radix publishes full blocks only; a
        partial tail block stays private — that is fine."""
        self.flush_to_kvc(chunk_tokens=chunk_tokens)
        self.manager.commit_prefix(self.seq)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def touch(self):
        """Bump last_access on every block (step/turn boundary fencing)."""
        if self.offset > 0:
            self.manager.touch(self.seq)

    def _head_dim(self):
        assert self._d is not None, "no tokens flushed yet; cannot attach empty cache"
        return self._d

    def detach(self):
        """Park: flush mirror KV into libKVC, then drop the mx mirrors."""
        if getattr(self, "detached", False):
            return
        self.flush_to_kvc()
        assert self.offset == self._mirror_offset(), (
            f"park incomplete: libKVC offset {self.offset} != "
            f"mirror offset {self._mirror_offset()}"
        )
        for lc in self.caches:
            lc._mirror.keys = None
            lc._mirror.values = None
            lc._mirror.offset = 0
        self.detached = True
        mx.clear_cache()

    def attach(self, timeout_ms=30000):
        """Gather bytes from libKVC (any tier) and rebuild mx mirrors."""
        if not getattr(self, "detached", False):
            return
        if self.offset == 0:
            self.detached = False
            return
        self._restore_mirrors_from_kvc(timeout_ms=timeout_ms)
        self.detached = False
        self.touch()

    def free(self):
        self.manager.free(self.seq)


def make_kvc_prompt_cache(model, manager, prompt_tokens=None,
                          n_kv_heads=None, head_dim=None):
    """Create a KvcPromptCache + layer caches for a model.

    Pass n_kv_heads and head_dim so radix-reused prefixes can restore mirrors
    before any flush has occurred (self._d would otherwise be None)."""
    coord = KvcPromptCache(model, manager, prompt_tokens,
                           n_kv_heads=n_kv_heads, head_dim=head_dim)
    return coord, coord.caches
