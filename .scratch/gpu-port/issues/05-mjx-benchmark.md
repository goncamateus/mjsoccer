# Benchmark bruto: MJX-JAX, MJX-Warp e CPU

Type: task
Status: resolved
Blocked by: 01, 04

## Pergunta

Nada a decidir: o 06 e o 09 precisam de números.

No env do 04, medir passos de física por segundo (SPS) e memória de GPU, sem lógica de task (só
`mjx.step` com ação aleatória ou zero), para:

| motor | modelo | cenas |
|---|---|---|
| MJX-Warp | base | 2 robôs, 7 robôs |
| MJX-JAX | variante sem mesh de robô: um casco convexo (ou primitivas) por robô para robô↔robô, robô↔campo e bola↔chassi, dribbler em cilindro (SDF) e em cápsula (descartável, só no script) | 2 robôs, 7 robôs |
| CPU | base | `mujoco.rollout` com 12 threads |

- Varredura de `nworld` (ex.: 256, 1024, 4096, 16384, até estourar os 12 GB).
- Para o MJX-Warp: `naconmax`/`njmax` mínimos sem overflow (`Data.overflow`) e o custo do graph mode
  (`WARP` vs `WARP_STAGED`) na ponte JAX.
- Itens da pesquisa 01 (§10): `max(ncon)`/`max(nefc)` na CPU por task com o controlador scriptado e colisões forçadas (com ação
  aleatória a instalação mediu `ncon` ≤ 11, `nefc` ≤ 42), para dimensionar
  `nconmax`/`naconmax`/`njmax`; memória do Warp por `nworld` (`testspeed --memory`); precisão do
  `sphere_cylinder` SDF do MJX-JAX × CPU no contato bola × barra (posição, normal, `dist`) contra a
  variante cápsula; iterações do Newton por passo no MJX-JAX com `tolerance` 1e-8 e 1e-6; MJX-JAX com
  `jax_enable_x64`; compile do MJX-Warp (kernels CCD, primeira chamada).
- Mesmas regras de memória do ticket de instalação (`MemoryMax`, nada em `/tmp`). O MJX-JAX no modelo
  base (e só com o corte chassi↔chassi) dá OOM de 8 GB no compile já com 2 robôs: não tentar.
  Partir de `research/04-smoke/smoke.py` (`cut_chassis(..., nomesh=True)`).
- Registrar o throughput em **decisões por segundo** também (24 passos de física por passo de
  controle, `repeat_action` = 1).

## Resultado esperado

Tabela de SPS × `nworld` × motor × cena, pico de memória, e as configurações usadas, em
`.scratch/gpu-port/research/05-mjx-benchmark.md`. Scripts ficam ao lado, não em `src/`.

## Answer

Resolvido em 2026-11-08. Detalhe, tabelas e método em
[`research/05-mjx-benchmark.md`](../research/05-mjx-benchmark.md); scripts e saída bruta em
[`research/05-bench/`](../research/05-bench/).

- **SPS de pico, layout aleatório** (passos de física/s somados; decisões/s = ÷ 24):
  MJX-Warp base **918 k** (2 robôs, 16 384 mundos, 9.5 GB) e **111 k** (7 robôs, satura em 1 024;
  OOM a 8 192); CPU `rollout` com 12 threads **245 k** e **80 k**; MJX-JAX na variante
  (prisma convexo por robô, CG) **6.4 k** e **1.7 k**.
- **Speedup sobre a CPU**: Warp 3.7× (2 robôs), 1.4× (7 robôs). MJX-JAX 40–50× **mais lento** que a CPU.
- **Warp**: `overflow` é máscara (512 = `njmax`, 4 = broadphase) e não para a simulação (só imprime avisos); o `naconmax` é um pool
  total dimensionado pelo broadphase (35/mundo e 120/mundo no aleatório, ~316/mundo em colisão
  forçada com 7 robôs; `nefc` até 184 → `njmax` 256). Cada slot custa ~7 KB de GPU: o pool de
  contatos limita o `nworld`. `WARP` > `NONE` > `WARP_STAGED` (111/103/94 k). Compile ~1.5 s
  (65 s com o cache de kernels frio). Newton: média 1–3 iterações por passo, máx 13.
- **MJX-JAX**: o compile é uma função por par de **meshes distintos**; meshes compartilhados entre os
  robôs fixam 61 grupos e ~47 s de compile. O Newton tem um penhasco a ≥ 256 mundos (~3.4 s por
  passo, causa não isolada; o CG não tem). `solver_niter` não é observável (volta 0). x64 é 6× mais
  lento. Cápsula × cilindro: mesmo SPS.
- **Contato bola × barra × CPU**: o Warp coincide (erro ~1e-5 mm, 0 faltas); o SDF de cilindro do
  MJX-JAX perde 1/3 dos contatos laterais e erra a posição em até 8 mm; a cápsula erra igual na CPU e
  no MJX e perde quase todo contato além das pontas.
- **CPU, dimensionamento**: `ncon` ≤ 15 e `nefc` ≤ 60 nas tasks com o controlador scriptado; 52 e 159
  em colisão forçada com 7 robôs.
