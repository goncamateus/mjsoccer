# 13: Script de treino `examples/ppo_mjx.py`

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `uv run --group gpu examples/ppo_mjx.py --env <id>` treina com o `brax ppo.train` vanilla (HPs como defaults de flag), espelhando o `examples/ppo_agent.py` (`--seed`, `--save`, `--play`). `--num-envs` default 8 192 (2 robôs) / 2 048 (7 robôs), revalidado na task real pela regra "maior estável 30 épocas com ≥ 2 GB de folga". `--graph-mode` default `WARP` (`WARP_STAGED` como opção) com `wp.clear_jax_callable_graph_cache()` + `gc.collect()` + `wp.synchronize()` no fim de cada época, no treino e na eval. `--repeat-action` repassado ao env, `action_repeat=1` no Brax. Stdout com passos, decisões/s, retorno de eval e taxas de `diverged`/`overflow`/`reset_overlap`; checkpoint Orbax em `runs/ppo_mjx_<env>/`. Eval embutido no MJX; eval standalone aborta se algum contador > 0; `--play` carrega o checkpoint e roda no backend CPU com o viewer.

**Blocked by:** 09 (Falhas por mundo no MJX-Warp), 10 (GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s)

**Status:** ready-for-agent

- [ ] Treino de 30 épocas em ContestedPossession e GoToBall com memória da GPU plana
- [ ] Defaults de `--num-envs` confirmados (ou corrigidos) na task real, com a folga anotada neste ticket
- [ ] Checkpoint salvo e `--play` roda a política no backend CPU
- [ ] Log mostra decisões/s e as taxas dos três contadores
