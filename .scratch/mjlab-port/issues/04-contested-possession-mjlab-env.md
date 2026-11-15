# 04: `mjsoccer.mjlab.make("ContestedPossession-v0")` + smoke

Spec: [backend mjlab](../spec.md) — Fase 2, Testing Decisions (seam 1)

**What to build:** o usuário cria a ContestedPossession no mjlab e passo em lote na GPU. Cena inteira via `SceneCfg.spec_fn` (modelo base exato, nomes do CPU, âncora vazia para o action term, `MujocoCfg` igual ao `<option>`); passo de disparo com `decimation = 48 × repeat_action` e `opt.timestep` por mundo (agenda 32 × dt/4 + 16 × dt ao disparar, 24 × dt + 24 × 0 senão); horizonte, recompensa e time-out nossos; pool de primeiros estados no `startup`; termination por overflow e por NaN com obs/recompensa zeradas; métricas `diverged`/`overflow`/`reset_overlap`. Usa a lógica do 03; os managers são cola fina.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] `make` lê `ENV_IDS`; mesmos espaços e layout de obs/ação do CPU; `repeat_action` com a mesma semântica
- [ ] Erro claro sem o grupo `mjlab` ou sem CUDA; `NotImplementedError` para id não portado
- [ ] Smoke (runner de subprocesso de GPU, pula sem CUDA): `make` + `reset` + ~10 passos, kick disparado, `diverged == overflow == 0`, obs finita
- [ ] Mundo ocioso bit a bit (`qpos`, `qvel`, `act`, `time`, `ctrl`) nas iterações a `timestep = 0`
- [ ] Lote misto com buffers subdimensionados em parte dos mundos: só esses terminam; o reset limpa só os bits deles
- [ ] Decisões/s ponta a ponta registradas contra o MJX-Warp (relatório)

## Comments

Hand-offs from 03 (`mjsoccer.mjlab.base`/`tasks`), left to the env:

- Per-world RNG: the logic takes `place`'s uniforms as an argument; the env owns the generator.
- `reset_overlap` on the first decision (`MjxEnv._first_decision`): `init_state` returns it in the reset metrics only.
- `seat_ball` (the pool of first states) and the firing schedule (`MjxEnv.timesteps`) are not in the logic.
- The task-logic parity runs in float64; the env will feed float32 from Warp. The controller's lateral sign at rest is rounding noise there, as on MJX.
