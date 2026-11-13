# 11: As quatro tasks restantes no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** GoToBallShoot, Dribbling, StaticDefenders e PassEndurance rodam no MJX-Warp, com os robôs scripted como no CPU (amarelos com `ctrl = 0`, companheiro do PassEndurance só com dribbler) e `vx, vy` zerados no PassEndurance. As 6 tasks ficam cobertas.

**Blocked by:** 10 (GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s)

**Status:** ready-for-agent

- [ ] `mjsoccer.mjx.make` funciona para todos os ids do `ENV_IDS`
- [ ] Task-logic parity e Warp smoke cobrem as 6 tasks
- [ ] `uv run pytest` verde
