# 06: Script de treino RSL-RL na ContestedPossession

Spec: [backend mjlab](../spec.md) — Fase 3

**What to build:** o pesquisador treina a ContestedPossession no mjlab com um script irmão do `ppo_mjx`, sobre o `MjlabOnPolicyRunner` (RSL-RL 5.5.2): mesmas flags e defaults de PPO; pool de primeiros estados refeito por época; log de decisões/s, retorno de eval e taxas de falha; checkpoint com config e política em ONNX. `--eval <dir>` avalia no mjlab e aborta com bits de overflow por tipo se algum mundo falhar; `--play <dir>` roda o ONNX (onnxruntime) no backend CPU, no viewer Studio, em processo próprio.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] `--env`, `--seed`, `--save`, `--play`, `--eval`, `--render`, `--repeat-action`, `--num-envs`, `--epochs`/`--timesteps`, `--num-eval-envs` e os hiperparâmetros PPO com os defaults do `ppo_mjx`
- [ ] ONNX exportado no fim e em cada checkpoint; a política ONNX roda no CPU com o mesmo espaço de obs/ação
- [ ] Smoke do treino: poucas iterações, checkpoint + ONNX gravados, `--eval` e `--play` headless carregam o resultado
