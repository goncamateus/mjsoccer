# 07: Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Grupo opcional `gpu` (`mujoco-mjx[warp]==4.0.0`, `jax[cuda12]`, `brax` no git `c860529`). Subpacote `mjsoccer.mjx` (`base`, `tasks`, `wrappers`) expõe `mjsoccer.mjx.make(id, **kw)` sobre o mesmo `ENV_IDS`, como `brax.envs.Env`, rodando no MJX-Warp sobre o modelo base, com `naconmax`/`njmax` dimensionados pelo env. ContestedPossession roda ponta a ponta: controlador de navegação e posse (sensor) em JAX, observação, recompensa, término, estado por mundo num pytree em `State.info` (`grace`, `steps`, chave RNG, estado da task); o env é dono de `steps`/`truncation`. Reset: `sample_pose` com 200 candidatos vetorizados e primeiro válido (fallback de maior folga), `seat_ball` com o mesmo loop físico. Constantes e config lidas do CPU, nada copiado. Kick fica para o ticket 08.

**Blocked by:** 03 (Posse pelo sensor `contact` no backend CPU), 04 (Ação em target pose e controlador de navegação no backend CPU)

**Status:** done

- [x] `import mjsoccer` não importa `jax` nem `mjsoccer.mjx`
- [x] `uv sync --group gpu` instala; `mjsoccer.mjx.make("ContestedPossession-v0")` faz `reset`/`step` sob `vmap` no Warp
- [x] Teste de task-logic parity (JAX-CPU, `JAX_PLATFORMS=cpu`, `importorskip`) cobre ContestedPossession: snapshots CPU seeds 0-4 via `put_data`, mesma obs/recompensa/término/posse/saída do controlador/estado por mundo em float32; `_place` com candidatos injetados dá as mesmas poses
- [x] Warp smoke de ContestedPossession (`make` + `reset` + ~10 passos, obs finita), pulado sem CUDA
- [x] `uv run pytest` verde com e sem o grupo `gpu`

## Comments

**Feito em 2026-11-14** (branch `gpu-port-07-mjx-skeleton`).

- **Grupo `gpu`** (`pyproject.toml`): `mujoco-mjx[warp]==4.0.0` (→ `warp-lang 1.18.0`),
  `jax[cuda12]>=0.12.0.1` (resolveu 0.12.0.1), `brax` no git `c860529` (reporta 0.14.2).
  `uv sync` sem o grupo remove tudo; `test_import_stays_light` (base) confere que
  `import mjsoccer` + `gym.make` não carregam `jax` nem `mjsoccer.mjx`.
- **Layout**: `src/mjsoccer/mjx/{__init__,base,tasks,wrappers}.py`. `make(id, *, repeat_action,
  graph_mode, randomize, nworld=1)`; task não portada → `NotImplementedError`; `randomize` ≠ None
  → `NotImplementedError` (ticket 14). `wrappers.py` está vazio (13-15).
- **Config do CPU**: o env instancia a classe CPU uma vez e lê `decimation`, `max_steps`,
  `_grace_steps`, `act_dim`, `observation_space`, `robots`/`ball` (mesmo `MjModel` em cache);
  `energy_steps`/`collision_speed` dos atributos de classe; os defaults do `seat_ball`
  (`distance`, `speed`, `settle`) pela assinatura do CPU. Constantes de `scene`/`base`/
  `navigation`/`tasks` importadas.
- **Seams de teste** (as do spec; sem usuário para confirmar): `mjsoccer.mjx.make` (smoke) e o
  snapshot → `put_data` (parity). Para testar a lógica sem física, o step é dividido em três
  métodos públicos: `control(state, action)` (controlador + dribbler → `ctrl`, `drive_cmd`),
  `physics(data)` (24 × `mjx.step`), `settle(state)` (posse, obs, recompensa, término, estado);
  o reset em `place(data, u)` (uniformes injetáveis, `n_uniform` por task) + `reset_physics`
  (`seat_ball`) + `init_state(data, rng)`. O parity põe a física do CPU entre `control` e `settle`.
- **Estado por mundo** em `State.info["world"]` (aninhado, não espalhado no `info`: o
  `EpisodeWrapper` do Brax usa `info["steps"]` próprio e somaria por cima): `grace`, `steps`
  (passos de controle), `drive_cmd`, `rng`, `prev`, `prev_goal`. `done` inclui a truncation em
  `max_steps`, marcada em `info["truncation"]` (só quando não terminou, como o Brax). `metrics`
  vazio até o ticket 09.
- **`naconmax`/`njmax`**: `naconmax = nworld × max(64, 24 × n_robots)` (64/mundo com 2 robôs,
  168 com 7: cobre os ~35/~120 candidatos medidos no 04), `njmax = 4 × isso`. O buffer de contato
  do Warp é do lote inteiro, então o env precisa do `nworld` (kwarg). O ticket 10 confere
  `overflow == 0` com 7 robôs.
- **`repeat_action` > 1**: `fori_loop` em que o mundo já terminado volta por `Data.where`. Um
  `while_loop` não serve: sob `vmap` o predicado por mundo faz o JAX batchear o buffer de contato
  compartilhado e o FFI do Warp recusa (`Leaf node leading dim ... does not match naconmax`).
- **`sample_pose`** (`mjx.base.sample_pose`): N candidatos de uma vez, primeiro livre por
  `argmax`, senão o de maior distância mínima + `overlap`. Nenhuma task usa ainda (no CPU o
  `sample_pose` também está sem uso; GoToBall/StaticDefenders usam `_scatter`, ≥ 0.2 m sem
  limite de tentativas — o ticket 10/11 decide se reaproveita com essa regra).
- **Achado — controlador em float32.** O eixo lateral do referencial rotacionado tem
  `to_target` identicamente 0; o `±MAX_DV` que o original soma ali com velocidade lateral ≈ 0
  pega o sinal do ruído de arredondamento (no CPU float64 também). Em float32 a zona ambígua
  vai a ~5e-4 m/s, e o sinal difere do CPU em ~1 passo de controle em 70 (sempre o passo 0, com
  o robô parado). Em float64 os dois batem em tudo. Por isso o task-logic parity roda com
  `jax_enable_x64` e compara na tolerância de float32 (1e-5). No Warp (float32) o jitter lateral
  do repouso continua lá, só com outro sinal — não é divergência de lógica, mas a paridade
  estatística (ticket 12) pode ver chatter um pouco diferente.
- **Testes novos**: `tests/test_mjx_task_logic.py` (18: `init_state`, `control`+`settle` por
  rollout de até 300 passos com ações aleatórias e depois a perseguição scripted até o término,
  `place` com candidatos injetados, `sample_pose` primeiro-livre e fallback, controlador em
  2 000 estados aleatórios; seeds 0-4; ~15 s). O snapshot usa `put_data` uma vez por modelo e
  depois troca os campos públicos do `Data` (o `put_data` inteiro custa 0.18 s por chamada).
  Mutação conferida: tirar o termo de energia derruba 5 testes. `tests/test_mjx_warp_smoke.py`
  (2: `repeat_action` 1 e 2, 4 mundos, 10 decisões, obs/qpos finitos, amarelo com a bola após o
  reset em todos os mundos) roda num subprocesso sem `JAX_PLATFORMS`, então não briga com o
  JAX-CPU do parity e solta a GPU ao sair; exit 77 → skip sem CUDA. Kick/chip ficam em repouso
  nos dois (ticket 08).
- **Verificação** (depois do merge com `gpu-port-impl`, ticket 05 dentro): `uv sync && uv run
  pytest` 95 passed, 2 skipped; `uv sync --group gpu` + `systemd-run --user --scope -p
  MemoryMax=10G uv run pytest` 115 passed. Nenhum processo na GPU depois.
