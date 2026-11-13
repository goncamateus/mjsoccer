# 07: Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Grupo opcional `gpu` (`mujoco-mjx[warp]==4.0.0`, `jax[cuda12]`, `brax` no git `c860529`). Subpacote `mjsoccer.mjx` (`base`, `tasks`, `wrappers`) expõe `mjsoccer.mjx.make(id, **kw)` sobre o mesmo `ENV_IDS`, como `brax.envs.Env`, rodando no MJX-Warp sobre o modelo base, com `naconmax`/`njmax` dimensionados pelo env. ContestedPossession roda ponta a ponta: controlador de navegação e posse (sensor) em JAX, observação, recompensa, término, estado por mundo num pytree em `State.info` (`grace`, `steps`, chave RNG, estado da task); o env é dono de `steps`/`truncation`. Reset: `sample_pose` com 200 candidatos vetorizados e primeiro válido (fallback de maior folga), `seat_ball` com o mesmo loop físico. Constantes e config lidas do CPU, nada copiado. Kick fica para o ticket 08.

**Blocked by:** 03 (Posse pelo sensor `contact` no backend CPU), 04 (Ação em target pose e controlador de navegação no backend CPU)

**Status:** ready-for-agent

- [ ] `import mjsoccer` não importa `jax` nem `mjsoccer.mjx`
- [ ] `uv sync --group gpu` instala; `mjsoccer.mjx.make("ContestedPossession-v0")` faz `reset`/`step` sob `vmap` no Warp
- [ ] Teste de task-logic parity (JAX-CPU, `JAX_PLATFORMS=cpu`, `importorskip`) cobre ContestedPossession: snapshots CPU seeds 0-4 via `put_data`, mesma obs/recompensa/término/posse/saída do controlador/estado por mundo em float32; `_place` com candidatos injetados dá as mesmas poses
- [ ] Warp smoke de ContestedPossession (`make` + `reset` + ~10 passos, obs finita), pulado sem CUDA
- [ ] `uv run pytest` verde com e sem o grupo `gpu`
