# 14: Domain randomization por mundo no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `mjsoccer.mjx.make(id, randomize=sampler)` com `sampler(mj_model, rng) -> None`. Para cada slot, uma vez no início do run: mutar o `MjModel` em cache, `mj_setConst`, copiar campos variados e derivados para uma pilha, restaurar o nominal; nunca copiar o `MjModel`. O diff contra o nominal vira o `in_axes`, adaptado ao `randomization_fn` do `brax ppo.train`; RNG derivado da seed de treino, determinístico por slot. Recusa sampler que mude `opt.timestep`; campos estruturais ficam fora. Docstring avisa que os contatos da bola são `<pair>`. `randomize=None` = modelo nominal. O `examples/ppo_mjx.py` avalia no nominal por default; `--eval-randomize` avalia sob DR. O CPU fica sem DR.

**Blocked by:** 13 (Script de treino `examples/ppo_mjx.py`)

**Status:** ready-for-agent

- [ ] Sampler que varia massa produz mundos com massa e `body_invweight0` coerentes por slot
- [ ] Mesma seed → mesmas variantes
- [ ] Sampler que muda `opt.timestep` levanta erro
- [ ] Treino com DR roda; `--eval-randomize` funciona; eval default no nominal
