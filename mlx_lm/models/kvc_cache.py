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

    def free(self):
        self.manager.free(self.seq)


def make_kvc_prompt_cache(model, manager, prompt_tokens=None):
    coord = KvcPromptCache(model, manager, prompt_tokens)
    return coord, coord.caches
