# 08: Kick no lote do MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Os mundos que disparam rodam o passo de disparo com 32 iterações a `dt/4` + 16 a `dt`; os ociosos rodam 24 a `dt` e ficam congelados com `timestep = 0` nas outras 24, via `opt.timestep` por mundo (`in_axes`), sem `Data.where` por passo. `plunger_lock` `int32 (n_robots, 2)` e `fired` no estado por mundo, decrementado por passo de controle, cooldown 10 lido do CPU. A `KICK_TABLE_MS` vem do CPU.

**Blocked by:** 07 (Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick)

**Status:** ready-for-agent

- [ ] ContestedPossession no MJX dispara kick e chip
- [ ] Task-logic parity cobre `plunger_lock`, `fired` e o cooldown
- [ ] Warp smoke: um mundo dispara e um ocioso mantém `qpos`/`qvel` bit a bit nas iterações a `timestep = 0`
