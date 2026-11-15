# 03 — `mjsoccer.mjlab.make("ContestedPossession-v0")`: decisões/s contra o MJX-Warp

Medido em 2026-11-14 (ticket 04), RTX 3060 12 GB, mujoco/mujoco-warp 4.0.1, os dois benches na
mesma sessão, sob `systemd-run --user --scope -p MemoryMax=10G`. **Relatório, não gate.**

- mjlab: [`../bench/dps_mjlab.py`](../bench/dps_mjlab.py) → [`results_mjlab.txt`](../bench/results_mjlab.txt).
  O env de verdade (`make` + `step()` do mjlab com os managers, pool de primeiros estados,
  auto-reset), ações uniformes no device, melhor de 3 × 10 s.
- MJX-Warp: `.scratch/gpu-port-impl/10-bench/dps_mjx.py` (Brax `training.wrap`) →
  [`results_mjx.txt`](../bench/results_mjx.txt).

| nworld | mjlab dec/s | MJX-Warp dec/s | pool (s) | Warp mempool (pico) | device usado |
|---:|---:|---:|---:|---:|---:|
| 1 024 | 13 272 | 13 442 | 0,54 | 99 MB | 998 MB |
| 4 096 | **16 598** | **16 537** | 1,84 | 339 MB | 2 600 MB |
| 8 192 | 17 230 | OOM (bench) | 3,56 | 659 MB | 4 688 MB |
| 16 384 | 17 534 | — | 7,05 | 1 298 MB | 8 952 MB |

- Mesmo empate do protótipo (16 588 a 4 096): o port com a lógica do 03 e os managers como
  cola não custou nada. Disparo em 17,4–17,6 % das decisões, 0 overflow em todos os tamanhos
  (`nconmax` 64, `njmax` 256).
- Memória do device a 16 384 caiu de 10,9 GB (protótipo) para 9,0 GB; o mempool do Warp segue
  1,3 GB. A diferença é o ticket 05.

## Achado: overflow de BROADPHASE marca todos os mundos

O buffer de contatos do Warp é um só para o lote (`naconmax = nworld × nconmax`). Com dois de
quatro mundos com robôs encavalados, o default (`nconmax` 64) estourou BROADPHASE e o bit veio
em **todos os quatro**, que terminaram juntos. Só o `njmax` (NEFC) é por mundo: com `nconmax`
512 e `njmax` 64, só os dois mundos encavalados estouraram e terminaram
(`tests/test_mjlab_smoke.py::test_mixed_batch_only_the_overflowed_worlds_end`). A história 8 do
spec ("só os mundos com bit terminam") vale por bit, e o BROADPHASE é posto em todo mundo.
