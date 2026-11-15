"""Where the device memory goes (ticket 05): one `make` per process, measured by source.

    systemd-run --user --scope -p MemoryMax=10G \
        uv run --group mjlab python .scratch/mjlab-port/bench/mem_mjlab.py <env_id> <nworld> [trim]

Stages: `ctx` (torch + Warp initialised on the device, nothing allocated), `make`, `pool`
(refill_pool + reset), `steps` (50 decisions). At each: device used (cuMemGetInfo), torch's
caching allocator (reserved/allocated), Warp's mempool (the device's default CUDA mempool,
reserved/used, driver attributes), and CUDA graph memory (cuDeviceGetGraphMemAttribute,
reserved/used). `rest` = device used - (ctx + torch reserved + mempool reserved + graph
reserved): module code, cuBLAS/cuRAND workspaces and the like. With `trim`,
cuDeviceGraphMemTrim after `make`. Prints one `RESULT {json}` line.
"""

import ctypes
import json
import sys

import torch
import warp as wp

MB = 2**20
cu = ctypes.CDLL("libcuda.so.1")
# CUmemPool_attribute / CUgraphMem_attribute
POOL_RESERVED, POOL_USED = 5, 7
GRAPH_USED, GRAPH_RESERVED = 0, 2


def _u64(fn, *args):
    v = ctypes.c_uint64()
    assert fn(*args, ctypes.byref(v)) == 0, fn
    return v.value


def snapshot() -> dict:
    torch.cuda.synchronize()
    wp.synchronize()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    assert cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) == 0
    pool = ctypes.c_void_p()
    assert cu.cuDeviceGetDefaultMemPool(ctypes.byref(pool), 0) == 0
    return {
        "device": (total.value - free.value) / MB,
        "torch_reserved": torch.cuda.memory_reserved() / MB,
        "torch_alloc": torch.cuda.memory_allocated() / MB,
        "pool_reserved": _u64(cu.cuMemPoolGetAttribute, pool, POOL_RESERVED) / MB,
        "pool_used": _u64(cu.cuMemPoolGetAttribute, pool, POOL_USED) / MB,
        "graph_reserved": _u64(cu.cuDeviceGetGraphMemAttribute, 0, GRAPH_RESERVED) / MB,
        "graph_used": _u64(cu.cuDeviceGetGraphMemAttribute, 0, GRAPH_USED) / MB,
    }


def main(env_id: str, nworld: int, trim: bool) -> dict:
    torch.zeros(1, device="cuda:0")
    wp.init()
    wp.zeros(1, device="cuda:0")
    import mjsoccer.mjlab

    out = {"ctx": snapshot()}
    env = mjsoccer.mjlab.make(env_id, nworld=nworld, seed=0)
    out["make"] = snapshot()
    if trim:
        assert cu.cuDeviceGraphMemTrim(0) == 0
        out["trim"] = snapshot()
    env.refill_pool()
    env.reset()
    out["pool"] = snapshot()
    g = torch.Generator(device=env.device).manual_seed(1)
    for _ in range(50):
        env.step(torch.rand(nworld, env.logic.action_size, device=env.device, generator=g) * 2 - 1)
    out["steps"] = snapshot()
    ctx = out["ctx"]["device"]
    for s in out.values():
        s["rest"] = s["device"] - ctx - s["torch_reserved"] - s["pool_reserved"] - s["graph_reserved"]
    return {"env": env_id, "nworld": nworld, "trim": trim,
            **{k: {f: round(v) for f, v in s.items()} for k, s in out.items()}}


if __name__ == "__main__":
    print("RESULT " + json.dumps(main(sys.argv[1], int(sys.argv[2]), "trim" in sys.argv[3:])),
          flush=True)
