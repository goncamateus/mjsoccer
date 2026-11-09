# Medir o kick no MJX-Warp: substeps, `timestep` por mundo e a tabela

Type: task
Status: resolved
Blocked by: —

## Pergunta

No CPU, o passo de controle em que um plunger dispara roda a `timestep / FIRING_SUBSTEPS`
(`src/mjsoccer/base.py:137-147`, `scene.py:70-74`) mudando o `MjModel` compartilhado; a
`KICK_TABLE_MS` foi medida assim. Num lote, cada mundo dispara numa hora. No env isolado
`~/.venvs/mjsoccer-gpu` (em disco; job com `MemoryMax`; sem mexer em `src/`, `models/`,
`pyproject.toml`), com bola assentada e robô parado como no `calibrate.py` do `physics-fidelity`:

1. **Velocidade de saída no MJX-Warp** (float32), kicker e chipper, nos pontos da tabela, em três
   modos: (a) `timestep` base, sem substep; (b) o lote todo a `timestep / 4` no passo de disparo;
   (c) `opt.timestep` por mundo via `in_axes` (só os mundos que disparam a `dt/4`). Contra a CPU
   (`mujoco 4.0.0`, float64, o mecanismo de hoje).
2. **(c) funciona?** O MJX-Warp aceita `opt.timestep` por mundo sem recompilar/recapturar? Como um
   lote anda 24 passos num mundo e 96 noutro — `while` mascarado, `Data.where` congelando, ou não dá.
   Efeito do `refsafe` (`timeconst` ≥ 2×`timestep`) nos contatos a `dt/4`.
3. **Custo**: decisões/s de (a), (b) e (c) num lote de 2 robôs a 4 096 mundos, com uma taxa de
   disparo realista (ex.: 1 em 20 decisões por mundo) e no pior caso (todo passo).

Registrar o erro por ponto da tabela contra o limiar do
[ticket de aceitação e paridade](09-acceptance-and-parity.md) (≤ 2 %).

## Answer

Resolvido em 2026-11-09. Detalhe e tabelas em
[`research/15-kick-on-warp.md`](../research/15-kick-on-warp.md); script e saída em
[`research/15-kick/`](../research/15-kick/).

- **Paridade**: o MJX-Warp (float32) reproduz o CPU 4.0 até 0.0006 % em todos os pontos, em todo
  esquema. Sem tabela float32 própria; a mesma tabela serve aos dois.
- **`timestep` por mundo** via `in_axes` funciona e é grátis (35.2 k × 35.0 k decisões/s). Congelar os
  mundos ociosos com `timestep = 0` é exato; o `Data.where` por passo custa −40 %.
- **Substep necessário**: K ≤ 16 passos a `dt/4` serrilha; K = 32 é o menor monótono; K ≥ 48 fica a
  ≤ 1 % do de hoje (K = 96).
- **Custo a 4 096 mundos (2 robôs)**: o mecanismo de hoje vira ~3.8× mais lento *sempre* (9.2 k), porque
  quase todo passo algum mundo dispara; K = 32 custa 2× (17.8 k); o **passo curto** (24 passos a `dt/4`,
  o passo de disparo dura 6.25 ms) custa ~nada (34.4 k), mas com curva e tabela próprias.
- **Tabela**: a de hoje (3.11, 60 Hz) fica 1–3 % acima do CPU 4.0 a 40 Hz; recalibrar no upgrade de
  qualquer jeito.
