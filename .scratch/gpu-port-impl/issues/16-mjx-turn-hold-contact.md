# 16: Posse no giro de 2 rad/s no MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** No giro de 2 rad/s com a bola (`hold_turn`), o MJX-Warp mantém a bola 69–78 de 120 passos (mediana 73), não determinístico do mesmo estado; a CPU mantém 78 sempre, limiar 74 (ticket 12, `## Comments`). Investigar o contato bola × dribbler em float32 no Warp (causa do não determinismo e da perda de posse) e corrigir só com física/solver legítimos, sem cheats. Decisão do usuário (2026-11-14): investigar antes de aceitar limiar próprio do MJX.

**Blocked by:** 12 (Paridade MJX-Warp × CPU)

**Status:** ready-for-agent

- [ ] Causa do não determinismo e da perda de posse identificada e registrada aqui
- [ ] MJX passa o limiar da CPU (74) no `hold_turn`, ou o usuário aceita um limiar MJX com os números medidos
- [ ] `xfail` do caso MJX removido; `uv run pytest -m validation` com o grupo `gpu` verde
