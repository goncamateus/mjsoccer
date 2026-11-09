# Brax `ppo.train` sobre o MJX-Warp nesta máquina (ticket 12)

Medido em 2026-11-08 na RTX 3060 12 GB, env `~/.venvs/mjsoccer-gpu`, cada run num processo com
`MemoryMax=8G`. Script e saída bruta em [`12-brax/`](12-brax/) (`brax_smoke.py`, `run.sh`,
`results.jsonl`, `progress.txt`). Fora de `src/`.

## Setup

- **Env de brinquedo** (`ToyChase`, `brax.envs.Env`): modelo base de 2 robôs (`bench.build(2)` do
  05), robô 0 persegue a bola (ação 3-d nos atuadores de velocidade, recompensa = −distância, `done`
  a < 12 cm), robô 1 parado. Um passo do env = uma decisão = 24 passos de física. Obs 7-d.
- **Wrapper**: `VmapWrapper` (ou `DomainRandomizationVmapWrapper` do Brax) → `EpisodeWrapper`
  (200 decisões) → auto-reset do Playground (estado inicial em cache) com o `where_done` trocado por
  `jax.vmap(lambda d, f, dn: d.where(dn, f))` (`mjx.Data.where` por mundo, `done` escalar dentro do
  `vmap`, como a docstring pede).
- **PPO**: `unroll_length` 20, `batch_size` = `nworld`/8, 32 minibatches, 4 updates/batch,
  128 envs de eval (instância própria do env), `normalize_observations`. `naconmax` 128/mundo,
  `njmax` 256 (com `njmax` 64 deu `nefc overflow` e recompensa NaN).
- **Contagem de capturas**: `warp.ScopedCapture` embrulhado antes de importar o MJX; conta toda
  captura feita pelo Warp (não as externas do JAX).

## 1. Versão do Brax

- **`brax==0.14.2` do PyPI não roda com `jax 0.12.0.1`**: `train.py:756` chama
  `jax.device_put_replicated`, removido no JAX 0.12 (`AttributeError: jax.device_put_replicated is
  deprecated; use jax.device_put instead`). Mesmo erro em `training/pmap.py`, `sac/`, `apg/`.
- O `main` do Brax já corrigiu (troca por `jax.device_put` com `NamedSharding`). **Pin**:
  `brax @ git+https://github.com/google/brax@c8605294264a7b7546268adbb4beabc9eb624000`
  (2026-10-20; versão interna ainda `0.14.2`). Até sair uma release > 0.14.2.
- Instalar o Brax não mexe em `mujoco`/`mujoco-mjx` 4.0.0, `jax` 0.12.0.1 nem `warp-lang` 1.18.0;
  adiciona `flax 0.12.12`, `optax 0.2.8`, `orbax-checkpoint 0.12.9`, `jaxopt 0.8.5`,
  `ml-collections`, `tensorboardx`, `flask` e afins.

## 2. `pmap` com 1 dispositivo

**Roda.** O `training_epoch` do Brax (`jax.pmap`) e o reset (`use_pmap_on_reset=True`, default) rodam
com as chamadas FFI do Warp dentro, numa GPU, em todos os runs. O fallback do 08 (vendorizar
`ppo/train.py` com `jit`) **não é necessário**. O `pmap` multi-dispositivo continua não testado
(fora de escopo).

## 3. SPS de treino ponta a ponta

`training/sps` do Brax = decisões/s do `training_epoch` (sem eval), na última época; "física" = ×24.
"Ponta a ponta" = decisões entre os dois últimos callbacks, eval incluído. Bruto = 05 (`WARP`).

| `nworld` | modo de grafo | treino (decisões/s) | física (SPS) | bruto do 05 | % do bruto | ponta a ponta | GPU (processo) | resultado |
|---|---|---|---|---|---|---|---|---|
| 1 024 | `WARP` | 24.4 k | 586 k | 645 k | 91 % | 20.5 k | 3.2 GB | ok (3 épocas) |
| 1 024 | `WARP_STAGED` | 13.2 k | 316 k | | 49 % | 10.9 k | 3.6 GB | ok |
| 4 096 | `WARP` | 31.3 k | 751 k | 855 k | 88 % | 29.5 k | 4.9 GB | **OOM na 6ª época** (run de 30) |
| 4 096 | `WARP_STAGED` | 16.4 k | 393 k | | 46 % | 13.0 k | 5.7 GB | **estável 30 épocas** |
| 4 096 | `NONE` | 17.8 k | 428 k | | 50 % | 5.6 k | 4.8 GB | estável 30 épocas; eval lenta |
| 8 192 | `WARP` | 31.0 k | 745 k | 901 k | 83 % | — | 9.5 GB | **OOM na 3ª época** |
| 8 192 | `WARP_STAGED` | — | — | | | — | — | **OOM no início** |
| 8 192 | `NONE` | 25.2 k | 604 k | | 67 % | 18.1 k | 9.5 GB | ok (3 épocas) |

- Com `WARP` o PPO custa ~10–17 % sobre a física bruta. O pico do JAX/XLA é pequeno (1.5 GB a 1 024,
  2.1 GB a 4 096, 3.0 GB a 8 192); o resto da GPU é do Warp (pool de contatos, temporários, grafos).
- Com `NONE` a eval (128 mundos × 200 decisões, sequencial) domina o ponta a ponta: ~50 s por eval
  a 4 096 contra ~25 s de treino.
- O treino aprende: retorno de eval −1 022 → ~−310 em 10 M decisões (4 096 mundos), parecido nos três
  modos.

## 4. Recaptura do grafo CUDA

- **`WARP` recaptura por época, não por passo.** O contador sobe +3 por época (2–3 depois de
  estabilizar), e é o mesmo com 4 ou 12 passos de treino por época (`ts=4` e `ts=12` a 1 024: 4 → 7 →
  10 → 11 nos dois). Dentro do `scan`/`fori_loop` o XLA reusa os buffers e o grafo é reaproveitado; a
  cada nova chamada do `training_epoch`/eval os endereços mudam e o Warp captura de novo.
- **A memória dessas capturas não volta.** Cada recaptura pede temporários novos ao pool do Warp
  (falha pedindo 759 MB a 8 192, 1.1 GB a 4 096), e o `nvidia-smi` mostra uso quase plano — o
  pool do Warp segura o que já pegou. Limitar o cache do Warp (`graph_cache_max=2`, via
  `functools.partial(wp.jax_callable, ...)`, o default é 32) **não resolve**: OOM na mesma época. Daí
  `WARP` mata qualquer treino longo nesta GPU (época 3 a 8 192, época 6 a 4 096); a 1 024 a mesma
  subida existe, só demora mais.
- **`WARP_STAGED` estabiliza**: 3 capturas no reset, 5 a partir da 1ª época, plano por 30 épocas,
  GPU plana (~5.7 GB a 4 096). Custo: metade do SPS de `WARP` e memória extra de staging (OOM a 8 192
  logo no início).
- **`NONE`**: zero capturas, estável; treino ~10 % acima do `WARP_STAGED`, eval ~4× mais lenta.
- Causa não isolada: se dá para ter endereços estáveis com `WARP` (doar buffers, uma chamada `jit` só
  para o treino inteiro) não foi testado.

## 5. DR por slot

- `randomization_fn(sys, rng)` → `(sys.replace(body_mass=mass_v), in_axes)` com
  `in_axes = jax.tree.map(lambda _: None, sys).replace(body_mass=0)` (massa da bola × U(0.8, 1.2) por
  slot, sem `mj_setConst` — só a mecânica do `in_axes`) roda no `DomainRandomizationVmapWrapper` do
  Brax sobre o `mjx.Model` do Warp, tanto em `WARP` quanto em `WARP_STAGED`.
- **Custo zero**: SPS com DR = sem DR dentro do ruído (1 024 `WARP`: 24.6 k × 24.4 k;
  `WARP_STAGED`: 13.0 k × 13.2 k; 8 192 `WARP`: 31.0 k × 31.0 k).
- **A DR não muda as capturas**: mesma contagem com e sem DR em todos os modos. O modelo em lote não
  causa recaptura extra; a recaptura do `WARP` é a da seção 4, com ou sem DR.
- O `DomainRandomizationVmapWrapper` do Brax troca `env.unwrapped.sys`; a variante do Playground troca
  `_mjx_model`. O env do lado MJX expõe `sys` e o vanilla serve.
