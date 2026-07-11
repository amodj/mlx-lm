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
        self.offset = 0  # tokens appended (or reserved) into libKVC
        self._h = None
        self._d = None
        self.detached = False
        self.caches = [KvcLayerCache(self, i) for i in range(self.n_layers)]
        # In-flight async pack: overlaps last-layer SDPA + lm_head.
        self._inflight_rec = None
        self._inflight_n = 0
        self._inflight_pos = 0

    def on_layer_update(self, idx, keys, values):
        assert keys.dtype == mx.float16, "libKVC v1 requires f16 KV"
        assert keys.shape[0] == 1, "libKVC v1 requires batch size 1"
        self._pending[idx] = (keys, values)
        if idx == self.n_layers - 1:
            # Schedule pack without blocking; commit lands before the next
            # flush / detach / free. Hides host materialization behind the
            # last-layer attention and lm_head that run after update_and_fetch.
            self._flush_async()

    def _pack_pending(self):
        """Build contiguous (T,L,2,H,D) f16 from pending layer updates."""
        pending = self._pending
        k = mx.stack([pending[i][0][0] for i in range(self.n_layers)])  # (L,H,T,D)
        v = mx.stack([pending[i][1][0] for i in range(self.n_layers)])
        k = mx.transpose(k, (2, 0, 1, 3))  # (T,L,H,D)
        v = mx.transpose(v, (2, 0, 1, 3))
        rec = mx.contiguous(mx.stack([k, v], axis=2))  # (T,L,2,H,D)
        n_new = int(rec.shape[0])
        H, D = int(rec.shape[3]), int(rec.shape[4])
        if self._h is None:
            self._h, self._d = H, D
        for i in range(self.n_layers):
            pending[i] = None
        return rec, n_new

    def _commit_inflight(self):
        """Sync any async pack and append into libKVC (ctypes loop, GIL released)."""
        rec = self._inflight_rec
        if rec is None:
            return
        host = np.array(rec, dtype=np.float16, copy=False)
        if not host.flags["C_CONTIGUOUS"]:
            host = np.ascontiguousarray(host)
        n_new = self._inflight_n
        assert host.nbytes == n_new * self.bpt
        self.manager.append_many(self.seq, host, self._inflight_pos)
        self._inflight_rec = None
        self._inflight_n = 0

    def _flush_async(self):
        self._commit_inflight()
        rec, n_new = self._pack_pending()
        self._inflight_rec = rec
        self._inflight_n = n_new
        self._inflight_pos = self.offset
        self.offset += n_new
        mx.async_eval(rec)
        # Do NOT touch_sequence here: append bumps last_access on the written
        # block. Full-sequence touch is O(n_blocks)/token and kills decode TpT.

    def _sync(self):
        """Drain async work; used by detach/attach/free/touch boundaries."""
        self._commit_inflight()
        if self._pending[self.n_layers - 1] is not None:
            # Rare: forward aborted mid-step; finish synchronously.
            self._flush_async()
            self._commit_inflight()

    def touch(self):
        """Bump last_access on every block (step/turn boundary fencing)."""
        self._sync()
        self.manager.touch(self.seq)

    def _head_dim(self):
        assert self._d is not None, "no tokens flushed yet; cannot attach empty cache"
        return self._d

    def detach(self):
        """Drop mx mirrors; bytes remain solely in libKVC."""
        if getattr(self, "detached", False):
            return
        self._sync()
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
        self._sync()
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
        self.touch()  # restored working set is in-use for this turn

    def free(self):
        self._sync()
        self.manager.free(self.seq)


def make_kvc_prompt_cache(model, manager, prompt_tokens=None):
    coord = KvcPromptCache(model, manager, prompt_tokens)
    return coord, coord.caches
