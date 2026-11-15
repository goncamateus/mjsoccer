# 03: Lógica da ContestedPossession em torch + task-logic parity

Spec: [backend mjlab](../spec.md) — Fase 1, Testing Decisions (seam 3)

**What to build:** a terceira implementação da lógica começa pela ContestedPossession, como funções torch puras sobre tensores do estado físico e do estado por mundo, sem tocar o sim do mjlab: controlador de navegação (quirks verbatim), kick/chip com `KICK_TABLE_MS` e trava do plunger, posse por sensor + grace, `_place`/`sample_pose` + `reset_overlap`, obs, recompensa e término da task. Constantes e config lidas do lado CPU. Um task-logic parity torch × CPU segura tudo, em torch-CPU, sem GPU. Port linha a linha do `mjsoccer.mjx`; o `Task` do protótipo é referência.

**Blocked by:** 02

**Status:** done

- [x] `import mjsoccer` não importa torch nem mjlab
- [x] Task-logic parity: snapshots CPU seeds 0-4, mesma ação, compara saída do controlador, obs, recompensa, término, posse e estado por mundo na tolerância de float32
- [x] Cobre `_place` com candidatos injetados e `plunger_lock`/`fired`/cooldown 10
- [x] Aborta se algum contador de falha for > 0; pula sem o grupo `mjlab`
