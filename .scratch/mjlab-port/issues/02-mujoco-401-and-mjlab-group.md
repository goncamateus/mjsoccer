# 02: mujoco 4.0.1 no `main`, grupo `mjlab` no lock, `scene.build` dividido

Spec: [backend mjlab](../spec.md) — Fase 0, Dependências

**What to build:** o `main` passa a rodar no mujoco 4.0.1 com a física intacta, e o terreno para o mjlab fica pronto: `uv sync --group mjlab` instala mjlab (git `6ab9846`), torch ≥ 2.14 cu130 e onnxruntime, ao lado do grupo `gpu`, no mesmo lock. O `scene.build` se divide em construir o `MjSpec` e compilar, sem mudar quem chama `build`. Referência: o branch `prototype/mjlab` (lock e divisão já feitos lá).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `mujoco`, `mujoco-mjx[warp]` em 4.0.1 (`mujoco-warp` 4.0.1, `warp-lang` 1.18.0)
- [ ] Grupo `mjlab` resolve no mesmo `uv lock` que o `gpu`, sem `conflicts`; torch e JAX importam no mesmo processo
- [ ] Construir o spec e compilar são passos separados; CPU e MJX inalterados
- [ ] Suíte padrão verde, inclusive os testes do grupo `rl` com o torch novo
- [ ] `-m validation` verde, inclusive paridade CPU↔MJX determinística, com seed e `hold_turn`, na 4.0.1
- [ ] Se algum critério cair: parar e reportar antes de seguir
