# Estabilizar o `WARP` e medir o treino de 7 robôs

Type: task
Status: open
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
