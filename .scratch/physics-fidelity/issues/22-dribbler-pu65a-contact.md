# Dar ao contato bola↔rolete a maciez do PU 65A

Type: task
Status: resolved
Blocked by: —

## Pergunta

Por que o robô perde a bola girando a ≥1 rad/s, e o que, dentro da física (sem junta, snap
cinemático ou restrição prendendo a bola, como grSim/rSim/Sumatra fazem), devolve a retenção?

## Diagnóstico

Sondas em `.scratch/physics-fidelity/prototype-dribbler-bar/` (scripts + `.txt`):

- A bola toca **só o rolete** (`trace_w1_a0.5_t0.txt`); no robô real também é o dribbler que
  amortece e conduz, então não há segundo contato a modelar.
- Não é geometria: barra em V (2/4 mm) e hélice dupla (as duas mãos) não seguram a 2 rad/s
  (`probe_tier0.json`). O `Dribblerschnecke.stl` é um tubo reto, não uma hélice.
- Não é atrito nem solver: μ 5/10, `cone=elliptic`, `impratio 10`, `noslip` não mudam nada
  (`knobs.py`). Nem rotação: 1500-9000 rpm, motor ×10, piora com rpm (`rpm_sweep_t0.txt`).
- **É o quique.** Parado, a normal bola↔rolete é bimodal (p10/p90 0.44/11 N) e o contato existe
  em 83 % dos substeps (`duty_t0.txt`): o `solref="0.004 0.5"` é subamortecido e ~10× mais duro
  que o PU (research `03` §5.5). Amortecer/amaciar o contato resolve (`damping_t0.txt`).

## Decisão

`solref="0.008 1"` nos dois pares bola↔rolete (`r_col_axisDribbler`, `r_col_barDribbler`) de
`scene.xml` e `scene_rollers.xml`. Fonte: rolete de PU Shore 65A (RobôCIn ETDP 2026 §2.2) →
módulo (Gent / Qi) → Hertz: ~0.2 mm de afundamento a 3 N → timeconst 7-9 ms com dampratio 1
(research `03` §5.5). O par `0.016 0.5`, também consistente com o afundamento, falha no tier 1
a 2 rad/s (15/120) e fica de fora (`pu65a.txt`).

## Resultado

`holds()` em 120 passos, giro no lugar, `dribble = 1`, pior das seeds 0-2 (`pu65a.txt`):

| | 0.5 | 1 | 2 | 3 rad/s |
|---|---|---|---|---|
| tier 0, antes → depois | 104 → 107 | 85 → 100 | 14 → 81 | 10 → 11 |
| tier 1, antes → depois | 94 → 103 | 82 → 96 | 13 → 73 | 11 → 11 |

Contato bruto parado: 74 → 97 % (tier 0), 65 → 98 % (tier 1). Posse parada
(`test_seated_ball_is_possessed`): 120/118/117/119 → 120/120/118/120. Fixado por
`test_seated_ball_survives_a_2_rad_s_turn`.

## Fica aberto

3 rad/s ainda derruba a bola nos dois tiers; as equipes publicadas seguram 3-7 rad/s por área
de contato (rolete macio/dividido), não por atrito (research `03` §1.3, §3.1). Um contato
esfera-cilindro do MuJoCo é um ponto só; o próximo passo físico seria área de contato
(`flexcomp` como luva, ou rolete dividido se o hardware for). O raio de colisão (9 mm) segue
maior que o real (6.5 mm): reduzir só o raio tira a bola do alcance do rolete, então pede
reposicionar o rolete junto.
