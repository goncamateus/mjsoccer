# Benchmark bruto: MJX-JAX, MJX-Warp e CPU

Type: task
Status: open
Blocked by: 01, 04

## Pergunta

Nada a decidir: o 06 e o 09 precisam de números.

No env do 04, medir passos de física por segundo (SPS) e memória de GPU, sem lógica de task (só
`mjx.step` com ação aleatória ou zero), para:

| motor | modelo | cenas |
|---|---|---|
| MJX-Warp | base | 2 robôs, 7 robôs |
| MJX-JAX | variante só com cápsulas | 2 robôs, 7 robôs |
| MJX-JAX | variante com cápsulas + um casco convexo por robô para robô↔robô e robô↔parede (descartável, só no script) | 2 robôs, 7 robôs |
| CPU | base | `mujoco.rollout` com 12 threads |

- Varredura de `nworld` (ex.: 256, 1024, 4096, 16384, até estourar os 12 GB).
- Para o MJX-Warp: `naconmax`/`njmax` mínimos sem overflow (`Data.overflow`) e o custo do graph mode
  (`WARP` vs `WARP_STAGED`) na ponte JAX.
- Registrar o throughput em **decisões por segundo** também (24 passos de física por passo de
  controle, `repeat_action` = 1).

## Resultado esperado

Tabela de SPS × `nworld` × motor × cena, pico de memória, e as configurações usadas, em
`.scratch/gpu-port/research/05-mjx-benchmark.md`. Scripts ficam ao lado, não em `src/`.
