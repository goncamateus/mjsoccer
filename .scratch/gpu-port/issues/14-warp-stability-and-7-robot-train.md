# Estabilizar o `WARP` e medir o treino de 7 robôs

Type: task
Status: resolved
Blocked by: 13

## Pergunta

O [ticket do modo de grafo](13-graph-mode-and-num-envs.md) fixou `WARP_STAGED` como default e a
regra do `--num-envs` (maior `nworld` estável por 30 épocas no modo default, ≥ 2 GB de folga na GPU).
Faltam dois números. No env isolado `~/.venvs/mjsoccer-gpu` (em disco; job com `MemoryMax`; sem
mexer em `src/`, `models/`, `pyproject.toml`), partindo de `research/12-brax/brax_smoke.py`:

1. **`WARP` estável?** Time-box de uma sessão. O `WARP` recaptura o grafo a cada chamada de
   `training_epoch`/eval e a memória não volta (OOM na época 6 a 4 096). Testar endereços de buffer
   estáveis: doação de buffers, um `jit` sobre o treino inteiro, liberar o pool do Warp entre
   capturas, ou outro mecanismo. Sucesso = contagem de capturas e GPU planas por 30 épocas a 4 096
   (2 robôs). Se der, medir o maior `nworld` de 2 robôs pela regra.
2. **Treino de 7 robôs**: `ppo.train` com `bench.build(7)` em `WARP_STAGED` (e em `WARP`, se o item
   1 der certo) a 512, 1 024 e 2 048 mundos; registrar decisões/s de treino e ponta a ponta, GPU,
   estabilidade por 30 épocas; aplicar a regra e dar o `--num-envs` default de 7 robôs.

## Answer

Resolvido em 2026-11-09. Detalhe e tabelas em
[`research/14-warp-stability-and-7-robot-train.md`](../research/14-warp-stability-and-7-robot-train.md);
script e saída bruta em [`research/14-warp/`](../research/14-warp/).

- **Causa**: o `WARP` só reaproveita um grafo com os mesmos endereços de buffer, e o XLA muda esses
  endereços a cada chamada do `training_epoch`/eval. Cada recaptura guarda os próprios temporários do
  Warp (o OOM é um `wp.empty` de 1.1 GB em `convex_narrowphase`, feito durante a captura). O cache
  segura os grafos velhos, e com `graph_cache_max=2` ainda são dois por callable.
- **Mecanismo**: `wp.clear_jax_callable_graph_cache()` com `gc.collect()` e `wp.synchronize()` no
  fim de cada época (`progress_fn`). GPU plana por 30 épocas a 4 096 e a 8 192, nenhum grafo
  velho vivo. A contagem de capturas **não** fica plana (+3 por época), e o SPS de treino cai 3–6 %
  em 30 épocas. Endereços estáveis de verdade não foram obtidos. Mexer no limiar do mempool não
  muda nada. `WARP_STAGED_EX` anda no nível do `WARP_STAGED` e sai.
- **Critério do 13**: a GPU fica plana, mas a contagem de capturas não. O dev decidiu que isso
  **passa**: `graph_mode` default `WARP` com a limpeza por época embutida no treino e na eval, e
  `WARP_STAGED` fica como flag.
- **2 robôs** (`WARP` + clear, 30 épocas): 4 096 → 27.4 k decisões/s ponta a ponta (2.1× o
  `WARP_STAGED`), 4.8 GB; 8 192 → 29.2 k, 9.55 GB, ~2.4 GB livres. Pela regra, **`--num-envs` 8 192**.
- **7 robôs** (30 épocas, todos estáveis): o treino satura em ~4.25 k decisões/s já a 512 mundos
  (92 % da física bruta). Ponta a ponta com `WARP` + clear: 3.5 / 3.8 / 4.0 k a 512 / 1 024 / 2 048
  (1.2× o `WARP_STAGED`), GPU 1.6 / 2.0 / 3.4 GB. **`--num-envs` 2 048** (maior medido; 4 096 não
  foi testado e, com o treino saturado, só ganharia o pouco que a eval dilui).
- **Ressalva**: a folga de 2.4 GB a 8 192 foi medida no env de brinquedo. A regra do 13 continua
  valendo para a task real.
