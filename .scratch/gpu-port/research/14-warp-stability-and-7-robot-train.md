# Estabilizar o `WARP` e o treino de 7 robôs (ticket 14)

Medido em 2026-11-08/09 na RTX 3060 12 GB, env `~/.venvs/mjsoccer-gpu` (`mujoco`/`mjx` 4.0.0,
`warp-lang` 1.18.0, `jax` 0.12.0.1, Brax `c860529`), cada run num processo com `MemoryMax` 8 GB
(12 GB nos de 7 robôs). Script e saída bruta em [`14-warp/`](14-warp/) (`train14.py`, `diag.sh`,
`phase2.sh`, `results.jsonl`, `progress.txt`, `log.txt`). Fora de `src/`.

Setup igual ao do [ticket 12](12-brax-ppo-warp-smoke.md) (env de brinquedo `ToyChase`, PPO do Brax,
`njmax` 256, `naconmax` 128/mundo, eval de 128 mundos). "Época" = uma chamada do `training_epoch`
mais a eval (`num_evals` 31 → 30 épocas). `train14.py` acrescenta `robots=`, `clear=1`, `relthr=` e
loga por época os grafos vivos no cache do Warp e o mempool do Warp.

## 1. Por que o `WARP` estoura

- **Chave da captura**: no modo `WARP`, `warp/_src/jax/ffi.py` só reaproveita um grafo se os
  endereços de **todos** os buffers de entrada e saída baterem com uma chamada anterior
  (`capture_key = hash((device, call_id, *ip, *op))`). O XLA dá endereços novos a cada chamada do
  `training_epoch`/eval, então cada época captura de novo (+3 por época).
- **Memória**: o OOM sai de `collision_convex.py:1320` (`convex_narrowphase`), um `wp.empty` de
  temporário feito **durante a captura** (1.14 GB a 4 096 mundos). Cada grafo capturado segura os
  próprios temporários; o mempool do Warp marca 0 MB o tempo todo, então a memória não está no pool.
  Com o cache default (32 por callable) os grafos velhos ficam vivos e acumulam até o OOM. O
  `graph_cache_max=2` do ticket 12 ainda guarda dois grafos por callable, e isso basta para estourar.
- **Correção**: ao fim de cada época (`progress_fn`), `wp.synchronize()`,
  `wp.clear_jax_callable_graph_cache()`, `gc.collect()` e `wp.synchronize()` de novo. Os grafos
  vivos ficam em 0 depois de cada época e a GPU fica plana. A recaptura continua (+3 por época), ao
  custo de ~0.5 s por época.

## 2. Diagnóstico (4 096 mundos, 2 robôs, 9 épocas)

| variante | resultado | treino (decisões/s) | ponta a ponta | GPU (processo) | capturas (vivas) |
|---|---|---|---|---|---|
| `WARP` | **OOM na 6ª época** (como no 12) | 31.3 k | — | 4.8 GB | 4 → 16 (16) |
| `WARP` + `clear=1` | ok | 30.7 k | 28.5 k | 4.8 GB, plana | 4 → 31 (0) |
| `WARP` + `clear=1` + `relthr=0` | ok, igual | 30.6 k | 28.4 k | 4.8 GB, plana | 4 → 31 (0) |
| `WARP_STAGED_EX` | ok | 15.9 k | 14.7 k | 5.7 GB | 3 → 5 |

- O limiar de devolução do mempool não muda nada, coerente com o pool em 0 MB: basta o `clear`.
- `WARP_STAGED_EX` (modo não testado no 12, cópias fora do grafo) fica no nível do `WARP_STAGED`
  (13.0 k ponta a ponta no 12): metade do `WARP`. Sai.

## 3. Regra do `--num-envs` com 2 robôs (`WARP` + clear, 30 épocas)

| `nworld` | ponta a ponta | treino, 1ª → última época | GPU (processo) | livre | retorno de eval final |
|---|---|---|---|---|---|
| 4 096 | 27.4 k | 31.2 k → 29.3 k | 4.8 GB, plana | ~7 GB | −267 |
| 8 192 | 29.2 k | 31.7 k → 30.3 k | 9.55 GB, plana | **~2.4 GB** | −268 |

- Os dois ficam estáveis por 30 épocas. O 8 192 passa na regra "≥ 2 GB de folga" por pouco
  (12 288 − 9 552 − ~330 de base ≈ 2.4 GB). Pela regra do [ticket 13](../issues/13-graph-mode-and-num-envs.md),
  o default de 2 robôs é **8 192**. O ganho sobre 4 096 é de 7 %.
- O SPS de treino cai 3–6 % ao longo das 30 épocas nos runs com `clear`; sem `clear` não há dado
  longo para comparar (o `WARP` morre antes). Custo provável da recaptura por época.
- Contra o `WARP_STAGED` do 12 (13.0 k ponta a ponta a 4 096): **2.1×**.

## 4. Treino de 7 robôs (30 épocas)

`bench.build(7)`, robô 0 persegue a bola, os outros 6 parados. Todos os seis runs estáveis, sem erro.

| `nworld` | modo | treino (decisões/s) | ponta a ponta | GPU (processo), plana | capturas | retorno final |
|---|---|---|---|---|---|---|
| 512 | `WARP_STAGED` | 3.26 k | 2.69 k | 2.1 GB | 5 | −1 244 |
| 512 | `WARP` + clear | 4.24 k | 3.53 k | 1.6 GB | 4 → 94 | −1 621 |
| 1 024 | `WARP_STAGED` | 3.45 k | 3.09 k | 2.7 GB | 5 | −1 213 |
| 1 024 | `WARP` + clear | 4.25 k | 3.84 k | 2.0 GB | 4 → 94 | −1 210 |
| 2 048 | `WARP_STAGED` | 3.53 k | 3.34 k | 4.5 GB | 5 | −1 211 |
| 2 048 | `WARP` + clear | 4.26 k | 4.04 k | 3.4 GB | 4 → 94 | −1 210 |

- **A GPU satura já em 512 mundos**: o treino fica em ~4.25 k decisões/s (≈ 102 k SPS de física,
  92 % dos 111 k brutos do 05 a 7 robôs). Mais mundos só diluem o custo fixo da eval no ponta a
  ponta (3.5 → 4.0 k).
- `WARP` + clear ganha **1.2×** do `WARP_STAGED` e gasta menos GPU.
- Maior `nworld` medido e estável: **2 048**, com ~8.5 GB livres. 4 096 não foi medido (seria
  ~2.7 h por run de 30 épocas). Com o treino já saturado, ele só renderia o pouco que a eval
  dilui.

## 5. O que entra no spec

- `graph_mode` default **`WARP`**, com limpeza do cache de grafos do Warp depois de cada chamada
  do `training_epoch` e da eval. No `examples/ppo_mjx.py`, isso fica dentro do `progress_fn`, que o
  Brax chama a cada época. O mesmo vale para a eval avulsa. `WARP_STAGED` continua disponível pela
  flag.
- `--num-envs` default: **8 192** com 2 robôs, **2 048** com 7 robôs.
- A folga de 2.4 GB a 8 192 foi medida no env de brinquedo (obs 7-d, rede default do Brax). Uma task
  real com obs e rede maiores pode passar do limite. A regra do 13 continua valendo para a task real.
