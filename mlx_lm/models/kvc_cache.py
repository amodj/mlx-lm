# Copyright © 2023-2024 Apple Inc.
# libKVC Stage A shadow cache — mirrors vanilla KVCache and shadows into libKVC.

import mlx.core as mx
import numpy as np
from .cache import KVCache, _BaseCache


class KvcLayerCache(_BaseCache):
    """Per-layer cache handed to the model; mirrors vanilla KVCache and
    reports every update to the KvcPromptCache coordinator."""

    def __init__(self, coord, layer_idx):
        self._coord = coord
        self._idx = layer_idx
        self._mirror = KVCache()

    def update_and_fetch(self, keys, values):
        out = self._mirror.update_and_fetch(keys, values)
        self._coord.on_layer_update(self._idx, keys, values)
        return out

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
    """Coordinator: owns one libKVC sequence; assembles token-major bytes
    from per-layer updates and appends once per token."""

    def __init__(self, model, manager, prompt_tokens=None):
        self.n_layers = len(model.layers)
        self.manager = manager
        self.seq = manager.allocate(list(prompt_tokens or []))
        self.bpt = manager.kv_bytes_per_token
        self._pending = [None] * self.n_layers
        self.offset = 0  # tokens appended into libKVC
        self._h = None
        self._d = None
        self.detached = False
        self.caches = [KvcLayerCache(self, i) for i in range(self.n_layers)]

    def on_layer_update(self, idx, keys, values):
        assert keys.dtype == mx.float16, "libKVC v1 requires f16 KV"
        assert keys.shape[0] == 1, "libKVC v1 requires batch size 1"
        self._pending[idx] = (keys, values)
        if idx == self.n_layers - 1:
            self._flush()

    def _flush(self):
        ks = [p[0] for p in self._pending]
        vs = [p[1] for p in self._pending]
        mx.eval(*ks, *vs)  # force lazy arrays
        n_new = ks[0].shape[2]
        H, D = ks[0].shape[1], ks[0].shape[3]
        if self._h is None:
            self._h, self._d = H, D
        # (L, H, T, D) -> token-major (T, L, 2, H, D), f16
        k = np.stack([np.array(x[0], copy=False) for x in ks])  # drop batch
        v = np.stack([np.array(x[0], copy=False) for x in vs])
        rec = np.empty((n_new, self.n_layers, 2, H, D), dtype=np.float16)
        rec[:, :, 0] = k.transpose(2, 0, 1, 3)
        rec[:, :, 1] = v.transpose(2, 0, 1, 3)
        raw = rec.tobytes()
        assert len(raw) == n_new * self.bpt
        for t in range(n_new):
            self.manager.append(
                self.seq,
                raw[t * self.bpt : (t + 1) * self.bpt],
                self.offset + t,
            )
        self.offset += n_new
        self._pending = [None] * self.n_layers
        self.manager.touch(self.seq)

    def _head_dim(self):
        assert self._d is not None, "no tokens flushed yet; cannot attach empty cache"
        return self._d

    def detach(self):
        """Drop mx mirrors; bytes remain solely in libKVC."""
        if getattr(self, "detached", False):
            return
        assert all(p is None for p in self._pending)
        for lc in self.caches:
            lc._mirror.keys = None
            lc._mirror.values = None
            lc._mirror.offset = 0
        self.detached = True
        mx.clear_cache()  # return freed buffers to the OS

    def attach(self, timeout_ms=30000):
        """Gather bytes from libKVC (promoting as needed) and rebuild mirrors."""
        if not getattr(self, "detached", False):
            return
        raw = self.manager.read(self.seq, timeout_ms=timeout_ms)  # valid prefix only
        T = self.offset
        assert len(raw) == T * self.bpt, (
            f"read length {len(raw)} != offset*bpt {T * self.bpt}"
        )
        if T == 0:
            self.detached = False
            return
        arr = np.frombuffer(raw, dtype=np.float16).reshape(
            T, self.n_layers, 2, -1, self._head_dim()
        )  # (T, L, 2, H, D)
        for l, lc in enumerate(self.caches):
            k = mx.array(np.ascontiguousarray(arr[:, l, 0].transpose(1, 0, 2)))[None]
            v = mx.array(np.ascontiguousarray(arr[:, l, 1].transpose(1, 0, 2)))[None]
            lc._mirror.state = (k, v)  # KVCache.state setter restores offset
        self.detached = False

    def free(self):
        self.manager.free(self.seq)


def make_kvc_prompt_cache(model, manager, prompt_tokens=None):
    coord = KvcPromptCache(model, manager, prompt_tokens)
    return coord, coord.caches
