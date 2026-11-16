# 12: Comparação RSL-RL × Brax PPO

Spec: [backend mjlab](../spec.md) — Fase 3

**What to build:** um relatório responde se treinar no mjlab é tão bom quanto no MJX: retorno de eval na ContestedPossession e na GoToBall, mesmo orçamento de decisões, mesmos hiperparâmetros de partida, seeds 0-4. Relatório, não gate.

**Blocked by:** 11

**Status:** done

- [x] Curvas de retorno de eval por decisões, mediana e [p25, p75] por backend, duas tasks, seeds 0-4
- [x] Relatório em `.scratch/mjlab-port/research/` com decisões/s de treino de cada um

## Comments

Done com escopo reduzido pelo usuário: só ContestedPossession, seed 0 (sem mediana/[p25, p75]).
Relatório: [research/07-rsl-rl-vs-brax.md](../research/07-rsl-rl-vs-brax.md); logs em
`bench/train12/`. 19,7 M decisões, 8 192 mundos: decisões/s de treino empatam (16,8k Brax,
16,2k mjlab); eval final −0,128 (Brax) × −0,018 (RSL-RL), nenhum aprende a task. Os dois
treinaram com episódios cortados a 80 decisões (`--timesteps` em 30 épocas + pool refeito por
época): igual para os dois, mas confunde o retorno. Ver o relatório para o rerun sugerido.
