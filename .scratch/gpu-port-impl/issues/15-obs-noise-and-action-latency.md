# 15: Ruído de observação e latência de ação no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Opções do wrapper de treino do lado MJX: ruído de observação (via `info["rng"]`) e latência de ação como buffer de N passos de controle no `State`. Desligadas por default, fora do task-logic parity, sem `actuator_delay` nativo.

**Blocked by:** 13 (Script de treino `examples/ppo_mjx.py`)

**Status:** done

- [x] Com as opções desligadas, comportamento idêntico ao de antes
- [x] Ruído ligado altera a obs de forma reprodutível por seed
- [x] Latência de N passos aplica a ação N passos de controle depois

## Comments

**Feito em 2026-11-14** (branch `gpu-port-15-noise-latency`).

- **Latência** no env (`mjsoccer.mjx.make(..., action_latency=N)`, default 0): conta passos de
  controle, que só o env vê (com `repeat_action > 1` um wrapper só veria decisões). O buffer é
  `info["world"]["action_queue"]` (N, action_size): a cada passo de controle aplica `queue[0]` e
  empurra a ação da decisão no fim. Como mora no `info["world"]`, o `AutoResetWrapper` do 13
  restaura o buffer junto (teste `latency_buffer_is_reset_per_world_by_the_auto_reset`). Começa
  com a ação "segurar a pose" de cada robô controlado (pose de reset, plungers e dribbler
  desligados), não com zeros (que mandariam o robô ao centro). Com N = 0 o `world` não ganha
  chave nenhuma. `step_raw` (cenários de envelope) não passa pelo buffer. Nada de
  `actuator_delay`.
- **Ruído** no wrapper de treino (`wrappers.wrap(..., obs_noise=sigma)`, default 0 = a mesma
  pilha de antes): `ObsNoiseWrapper`, o mais externo, soma N(0, sigma²) à obs normalizada, no
  reset e em cada passo (inclusive a primeira obs depois de um auto-reset), com chaves em
  `info["rng"]` por mundo, derivadas da chave do reset por `fold_in` (não mexe nos sorteios do
  env; mesma seed, mesmo ruído). A física não vê o ruído.
- **`ppo_mjx.py`**: `--obs-noise` e `--action-latency` (default 0); valem no treino e no eval
  embutido do Brax; `--eval`/`--play` rodam sem eles.
- **Achado**: o MJX-Warp não é reprodutível bit a bit: dois rollouts idênticos diferem ~5e-7 já no
  reset (o `seat_ball`) e ~1e-4 em 5 decisões (atomics). Os testes de "idêntico" usam
  `atol = 2e-3` (o ruído testado tem std 0.05); o "desligado" usa 1e-2 com a ação "segurar" (com
  repeat 2 e ações aleatórias o jitter passou 2e-3 em 2 de 6 rodadas) e compara a pilha de wrappers
  e as chaves de `info`/`info["world"]`, que são exatas. O arquivo não importa o `jax` na coleta
  (`find_spec`), senão o parity, coletado depois, cairia na GPU (o achado do 13).
- **Testes** (`tests/test_mjx_noise_latency.py`, subprocesso na GPU, 6): desligado = antes
  (repeat 1/2); kick pedido na decisão 0 dispara no passo de controle N (lock lido decisão a
  decisão, repeat 1/2, N = 3) e o robô fica parado até lá; buffer restaurado pelo auto-reset;
  ruído reprodutível por seed, diferente entre seeds, std ~sigma, física igual.
  `tests/test_ppo_mjx.py` ganhou os defaults das duas flags.
- **Treino**: `ppo_mjx.py --env ContestedPossession-v0 --num-envs 256 --num-eval-envs 16
  --epochs 1 --obs-noise 0.02 --action-latency 2` rodou limpo (20 480 decisões, contadores 0,
  748 MB de GPU; a GPU dividida com outros dois processos).
- **Verificação** (`git merge gpu-port-impl`: já em dia): `--group gpu` 240 passed + 1 falha do
  `test_viewer` web que passa sozinho (porta disputada com outro worktree, imagino); sem o grupo 99
  passed, 4 skipped. Nenhum processo meu na GPU.
