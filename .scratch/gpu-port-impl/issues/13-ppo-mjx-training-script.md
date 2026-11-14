# 13: Script de treino `examples/ppo_mjx.py`

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `uv run --group gpu examples/ppo_mjx.py --env <id>` treina com o `brax ppo.train` vanilla (HPs como defaults de flag), espelhando o `examples/ppo_agent.py` (`--seed`, `--save`, `--play`). `--num-envs` default 8 192 (2 robôs) / 2 048 (7 robôs), revalidado na task real pela regra "maior estável 30 épocas com ≥ 2 GB de folga". `--graph-mode` default `WARP` (`WARP_STAGED` como opção) com `wp.clear_jax_callable_graph_cache()` + `gc.collect()` + `wp.synchronize()` no fim de cada época, no treino e na eval. `--repeat-action` repassado ao env, `action_repeat=1` no Brax. Stdout com passos, decisões/s, retorno de eval e taxas de `diverged`/`overflow`/`reset_overlap`; checkpoint Orbax em `runs/ppo_mjx_<env>/`. Eval embutido no MJX; eval standalone aborta se algum contador > 0; `--play` carrega o checkpoint e roda no backend CPU com o viewer.

**Blocked by:** 09 (Falhas por mundo no MJX-Warp), 10 (GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s)

**Status:** done

- [x] Treino de 30 épocas em ContestedPossession e GoToBall com memória da GPU plana
- [x] Defaults de `--num-envs` confirmados (ou corrigidos) na task real, com a folga anotada neste ticket
- [x] Checkpoint salvo e `--play` roda a política no backend CPU
- [x] Log mostra decisões/s e as taxas dos três contadores

## Comments

**Feito em 2026-11-14** (branch `gpu-port-13-ppo-mjx`).

- **Script** `examples/ppo_mjx.py`: `brax ppo.train` vanilla com `wrap_env_fn=mjsoccer.mjx.wrappers.wrap`,
  `episode_length = ceil(max_steps / repeat_action)`, `action_repeat=1` (o `wrap` recusa outro valor).
  Flags: `--env --seed --save --play --eval --episodes --render --repeat-action --num-envs
  --graph-mode {WARP,WARP_STAGED} --epochs 30 --timesteps --num-eval-envs 128` + os HPs
  (lr 3e-4, entropy 1e-2, gamma 0.99, gae 0.95, clip 0.2, unroll 20, 32 minibatches, 4 updates,
  `batch_size = num_envs // 8`, grad norm 0.5, obs normalizadas). Default: uma iteração PPO por
  época (4 unrolls de todos os mundos). `XLA_PYTHON_CLIENT_PREALLOCATE=false` por `setdefault`.
  Checkpoint Orbax do próprio Brax (`runs/ppo_mjx_<env>/<step:012d>/`) + `args.json`. `progress_fn`
  imprime passos, decisões/s (relógio entre épocas, e o `training/sps` do Brax), retorno e duração
  do eval, taxas por episódio de eval de `diverged`/`overflow`/`reset_overlap`, MB da GPU do
  processo e livres; no `WARP` faz `wp.synchronize` + `clear_jax_callable_graph_cache` + `gc` +
  `wp.synchronize` (o `progress_fn` vem depois do treino e do eval da época). `--eval` roda a
  política uma vez por mundo no MJX e aborta (`check_failures` + `reset_overlap`) a cada passo.
  `--play` força `JAX_PLATFORMS=cpu` e roda `gymnasium.make` com o viewer Studio.
- **Problema 1, auto-reset** (`mjx/wrappers.py`): `AutoResetWrapper` próprio guarda e restaura
  `info["world"]` junto com `pipeline_state` (por `mjx.Data.where` vmapeado) e `obs`. **Achado
  extra**: o `EpisodeWrapper` do Brax escreve `truncation = 1 - done` no passo do timeout, então o
  timeout do próprio env (done, truncation 1) saía como truncation 0 = terminal, sem bootstrap.
  O `EpisodeWrapper` daqui mantém a truncation do env e soma a dele. Teste novo
  `test_timeout_is_truncation_and_resets_per_world_state` (6 tasks × repeat 1/2, horizonte 0.1 s):
  com `training.wrap` do Brax falhava nas duas coisas; e o `test_nan_world...` agora anda 3
  decisões antes do NaN e exige o `info["world"]` do mundo 0 bit a bit igual ao primeiro, e os
  outros mundos intactos.
- **Problema 2, `repeat_action > 1`**: não era o `scan` do `EpisodeWrapper`. O `fori_loop` do
  `step` fazia `jnp.where` em todo o `State` menos o `pipeline_state` — inclusive no
  `info["first_pipeline_state"]` do auto-reset, um `Data` do Warp cujo buffer de contato é do lote
  inteiro; o `where` batcheava esse buffer e o tracer vazava do `vmap`. Reproduz com
  `jit(vmap(step))` puro e um `Data` qualquer em `info` (`13-ppo/repro_tracer.py vmap_info`).
  Agora só os campos do env (`obs`, `done`, `metrics`, `info["world"]`, `info["truncation"]`) são
  mesclados. TODO do smoke removido; o teste do NaN roda com repeat 1 e 2. Treino de 1 época do
  GoToBall com `--repeat-action 2` rodou limpo.
- **`reset_overlap` no log**: os somatórios do Brax (episódio e eval) só contam métricas de passo,
  nunca as do estado de reset. O env guarda `world["overlap"]` e a primeira decisão do episódio
  repete o flag (`test_reset_overlap_is_counted_on_the_first_decision`). As taxas do log são do
  eval; as do treino não aparecem sem `log_training_metrics` (callback por passo, não liguei).
- **Dois bugs do Brax no checkpoint** (commit pinado): `load_policy` dá `KeyError: None` (o factory
  default salva `mean_kernel_init_fn: null`) e `KeyError: 'state'` (o `ppo.train` salva
  `observation_size` como `specs.Array`). O `ppo_mjx.load_policy` contorna os dois. O Orbax também
  exige caminho absoluto.
- **Testes** (`tests/test_ppo_mjx.py`): defaults de `--num-envs` por robôs, `--graph-mode`,
  overrides, `action_repeat` 1 e a regra `batch*minibatches % num_envs`; `--play` num checkpoint
  feito como o `ppo.train` faz, num subprocesso JAX-CPU (carrega o último passo, `backend cpu`,
  completa 1 episódio). Os de CLI rodam sem o grupo `gpu`.
- **`--num-envs`** (um processo por tamanho, 3 épocas, `13-ppo/probe.out`; os 3 primeiros com 1
  processo do ticket 11 na GPU, ~0.5 GB, que deixou o dps ~40 % menor mas não mexe na memória):

  | task | num_envs | GPU do processo | livre | |
  |---|---|---|---|---|
  | ContestedPossession (2) | 4 096 | 3.36 GB | 8.1 GB | |
  | | **8 192** | 6.17 GB | **5.2 GB** | default |
  | GoToBall (7) | **2 048** | 5.16 GB | **6.6 GB** | default |
  | | 4 096 | 9.67 GB | 1.7-2.3 GB | < 2 GB: reprovado |

  16 384 com 2 robôs não cabe (~12 GB por extrapolação). Defaults confirmados: 8 192 / 2 048.
- **Treino de 30 épocas** (GPU sozinha, `13-ppo/train30_*.log`, RTX 3060):
  - ContestedPossession, 8 192 mundos: 19.7 M decisões em 37 min; treino 16.0-16.4 k decisões/s
    (o bench do 10 deu 16.4 k), 9.0 k contando o eval. GPU **6 168 MB plana da época 2 à 30**
    (5 970 na 0), 5.87 GB livres. Retorno do eval −0.03 → +0.16; diverged/overflow/reset_overlap
    0 em todas as épocas.
  - GoToBall, 2 048 mundos: 4.9 M decisões em 87 min; treino 2.15 k decisões/s (bench 2.2 k),
    0.96 k com o eval (o eval de 128 mundos × 1 200 decisões domina a época). GPU **5 156 MB plana
    da época 3 à 30**, 6.9 GB livres. Retorno 0 → 0.055 (fração de episódios que chegam à bola);
    contadores 0.
  - Sem tuning (fora do escopo): o retorno sobe, só isso foi pedido.
- **`--eval`** standalone nos dois checkpoints finais: ContestedPossession −0.094 ± 0.060 (223
  decisões), GoToBall 0.094 ± 0.291 (1 131); "no failures". Difere do eval embutido porque o Brax
  avalia com política estocástica (`deterministic_eval=False`) e o `--eval` é determinístico.
- **`--play`** nos dois: backend cpu, headless e com o viewer Studio (1 episódio do
  ContestedPossession com janela, 297 decisões).
- **Achado (não mexi)**: rodar `pytest tests/test_mjx_warp_smoke.py tests/test_mjx_task_logic.py`
  nessa ordem falha 4 rollouts do parity do GoToBall (`ctrl` no passo 0, sinal lateral): o smoke
  importa o `jax` antes de o parity setar `JAX_PLATFORMS=cpu`, então o parity roda na GPU
  (float32/TF32 no `xmat @ v`). Na ordem default passa.
- **Verificação** (depois de `git merge gpu-port-impl`, 11 e 12 dentro; um conflito no `main` do
  smoke): `uv sync --group gpu` + `systemd-run --user --scope -p MemoryMax=10G uv run pytest`
  234 passed; `uv sync && uv run pytest` 98 passed, 3 skipped. Nenhum processo na GPU depois.
  Nenhum artefato de `runs/` no git.
