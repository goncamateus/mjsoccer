# 11: `--num-envs` default por cena (treino longo)

Spec: [backend mjlab](../spec.md) — Fase 3

**What to build:** o default do `--num-envs` do script mjlab é o maior `nworld` estável num treino longo (não os ~40 s do protótipo), medido para a cena de 2 robôs e a de 7, com 0 overflow e memória plana.

**Blocked by:** 05, 06, 08

**Status:** ready-for-agent

- [ ] Treino longo medido com 2 e 7 robôs em candidatos de `nworld` (incluindo 8 192 e 16 384 com 2 robôs)
- [ ] Default gravado no script por cena, com a medição registrada
