# 06: Script de treino RSL-RL na ContestedPossession

Spec: [backend mjlab](../spec.md) — Fase 3

**What to build:** o pesquisador treina a ContestedPossession no mjlab com um script irmão do `ppo_mjx`, sobre o `MjlabOnPolicyRunner` (RSL-RL 5.5.2): mesmas flags e defaults de PPO; pool de primeiros estados refeito por época; log de decisões/s, retorno de eval e taxas de falha; checkpoint com config e política em ONNX. `--eval <dir>` avalia no mjlab e aborta com bits de overflow por tipo se algum mundo falhar; `--play <dir>` roda o ONNX (onnxruntime) no backend CPU, no viewer Studio, em processo próprio.

**Blocked by:** 04

**Status:** done

- [x] `--env`, `--seed`, `--save`, `--play`, `--eval`, `--render`, `--repeat-action`, `--num-envs`, `--epochs`/`--timesteps`, `--num-eval-envs` e os hiperparâmetros PPO com os defaults do `ppo_mjx`
- [x] ONNX exportado no fim e em cada checkpoint; a política ONNX roda no CPU com o mesmo espaço de obs/ação
- [x] Smoke do treino: poucas iterações, checkpoint + ONNX gravados, `--eval` e `--play` headless carregam o resultado

## Comments

Done (ticket 06): `examples/ppo_mjlab.py`, tests in `tests/test_ppo_mjlab.py`.

Decisions and deviations, for the spec owner:

- The flags of ticket 10 (`--obs-noise`, `--action-latency`, `--randomize`, `--eval-randomize`)
  are not here yet; `--num-envs` defaults to ppo_mjx's batch until ticket 11. `--env` defaults
  to ContestedPossession (GoToBall is not ported).
- ppo_mjx's PPO mapped onto RSL-RL: `--unroll-length` -> `num_steps_per_env`,
  `--num-updates-per-batch` -> learning epochs, `value_loss_coef` 0.25 (Brax's 0.5 * 0.5),
  Brax's networks (4 x 32 / 5 x 256, swish, obs normalization), fixed LR, no value clipping.
  Dropped: `--batch-size` (an RSL-RL iteration is one unroll of every world), `--reward-scaling`,
  the Brax variants; the Gaussian policy is clipped to [-1, 1] instead of tanh-squashed.
- An epoch: `refill_pool()` + reset, `learn()` for one episode's worth of iterations (or
  `--timesteps` split), eval on `--num-eval-envs` worlds, checkpoint + ONNX. The ONNX has a
  dynamic batch axis (mjlab's export fixes it at 1) and a failed export aborts (mjlab's warns).
- `--eval` runs the ONNX (onnxruntime, CPU) on mjlab with `auto_reset` off, so Warp's overflow
  bits are read before a reset clears them.
- The smoke test checks CUDA in-process (`torch.cuda.is_available()`, no context) instead of a
  `SKIP` exit from the child, since the child is the user's script.
