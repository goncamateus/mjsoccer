# 05 — Benchmark bruto: MJX-Warp, MJX-JAX e CPU

Medido em 2026-11-08 nesta máquina (RTX 3060 12 GB, Ryzen 5 5600X 6c/12t, 30 GB), no env do 04
(`~/.venvs/mjsoccer-gpu`: `mujoco`/`mujoco-mjx 4.0.0`, `warp-lang 1.18.0`, `jax 0.12.0.1`).
Scripts em [`05-bench/`](05-bench/): `bench.py` (um estágio por processo), `run.sh` (cgroup de 8 GB,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`), `sweep.sh`/`sweep2.sh` (as varreduras), `probe.py` (isola o
compile e o passo do MJX-JAX). Saída bruta: `sweep1.out`, `sweep2.out`, `results*.jsonl`, `log_condensed.txt`.

## Como se mediu

- **Só física.** `mjx.step` (ou `mujoco.rollout`), sem task. Um passo de controle = 24 passos de
  física (40 Hz a 1/960 s); a ação é reamostrada a cada passo de controle. **SPS** = passos de física
  por segundo somados sobre os mundos; **DPS** (decisões/s, `repeat_action` = 1) = SPS / 24.
- **Layout `random`**: cada robô (e a bola) numa faixa de x de 1.5 m, y e yaw sorteados; comandos de
  velocidade uniformes no `ctrlrange`, dribbler ligado, kick/chip 0. **`cluster`** (colisões
  forçadas): robôs num anel de 0.30–0.40 m em volta da bola, todos perseguindo-a a 1.5 m/s.
- MJX: `jit(scan(decisões, fori(24, vmap(mjx.step))))`, compile por `lower().compile()`, depois
  chamadas cronometradas até ≥ 5 s. MJX-Warp com `graph_mode=WARP` salvo indicação.
- **Modelo base** no Warp e na CPU. **Variante descartável** no MJX-JAX (só no script, via `MjSpec`):
  os 87 meshes de colisão do robô saem; entra **um prisma convexo** de 16 lados por robô (pontos de
  suporte dos meshes reais, extrudados no z deles: 30 vértices, frente chata preservada) para
  robô↔robô, robô↔campo e bola↔chassi; as placas do kicker/chipper (16 meshes) colidem só com a bola,
  por `<pair>` explícito. Massas e inércias conferidas iguais às do base. Dribbler em cilindro (SDF) ou
  cápsula (`drib=cap`, mesmo raio, mesmo comprimento total).
- O MJX-JAX com Newton tem um penhasco (abaixo): as varreduras dele usam **CG** (`solver=CG`, numérica,
  restrição 7) e chamadas de 1 decisão.

## Resultado principal: SPS × `nworld` × motor (layout `random`)

| motor | modelo | robôs | 256 | 1 024 | 4 096 | 8 192 | 16 384 | teto |
|---|---|---|---|---|---|---|---|---|
| MJX-Warp | base | 2 | 330 k | 645 k | 855 k | 901 k | **918 k** (9.5 GB) | OOM a 32 768 |
| MJX-Warp | base | 7 | 96 k | **111 k** | 111 k (5.9 GB) | OOM | — | satura em 1 024 |
| MJX-JAX (CG) | variante | 2 | 5.6 k | **6.4 k** | 6.3 k (4.3 GB) | — | OOM (pede 6.9 GB) | satura em 1 024 |
| MJX-JAX (CG) | variante | 7 | 1.66 k | **1.70 k** (4.4 GB) | falha no autotune | — | — | |
| CPU, 12 threads | base | 2 | | 241 k | **245 k** | | | |
| CPU, 12 threads | base | 7 | | 79 k | **80 k** | | | |
| CPU, 1 thread | base | 2 / 7 | 43.5 k / 14.0 k | | | | | |

Colunas da CPU = tamanho do lote do `rollout`. Em **decisões/s** (÷ 24), no melhor ponto:

| | 2 robôs | 7 robôs |
|---|---|---|
| MJX-Warp | 38 k | 4.6 k |
| CPU 12 threads | 10 k | 3.3 k |
| MJX-JAX (variante, CG) | 270 | 71 |

**Speedup sobre a CPU com 12 threads**: MJX-Warp **3.7×** (2 robôs) e **1.4×** (7 robôs);
MJX-JAX **0.026×** e **0.021×** — 40–50× *mais lento* que a CPU, mesmo com a variante simplificada.

Colisões forçadas (`cluster`, buffers que não estouram): Warp 588 k SPS (2 robôs, 4 096 mundos) e
52 k (7 robôs, 1 024 mundos, `naconmax` 640/mundo, `njmax` 256); MJX-JAX 11 k (2 robôs, 4 096).

## MJX-Warp

- **Buffers.** `overflow` é uma máscara: **512 = `njmax`**, **4 = broadphase** (`naconmax`). O
  `nacon` real é ~1/mundo (`random`) e ~12/mundo (`cluster`, 7 robôs); o que dimensiona o
  `naconmax` são os **candidatos do broadphase** (`ncollision`, um pool **total** sobre os mundos):

  | cena | `ncollision` pico/mundo | `nefc` máx/mundo | mínimo sem overflow usado |
  |---|---|---|---|
  | 2 robôs `random` | 35 | ≤ 46 | 64 / 64 |
  | 2 robôs `cluster` | 58 | 32 | 64 / 64 |
  | 7 robôs `random` | 120 (constante) | 20 | 128 / 64 |
  | 7 robôs `cluster` | ~316 | 184 | 640 / 256 (128/256 ainda dá broadphase overflow) |

  A primeira passada do `cluster` de 7 robôs (128/64) **estourou** (`overflow` 519) e rodou até o fim:
  o Warp imprime avisos (`nefc overflow - please increase njmax beyond 64`, `njmax_nnz overflow`,
  `solver iterations limit reached`, desligáveis por `m.opt.warn_overflow`) mas não para nem marca o
  resultado. Quem consome precisa ler `_impl.overflow`.
- **Memória.** Domina o pool de contatos: **~7 KB de GPU por slot de `naconmax`** (192 vs 128
  slots/mundo a 1 024 mundos: +480 MB; 640/mundo: 5.3 GB). O `njmax` 64 → 256 custa ~nada. Por isso
  o teto de 7 robôs é ~4 096 mundos (8 192 × 136 slots dá OOM ao criar o grafo CUDA) e o de 2 robôs
  ~16 384. Todo OOM do Warp aparece como `Graph creation error: Warp CUDA error 2: out of memory`.
  RSS do host 1.8–3 GB.
- **Graph mode** (7 robôs, 4 096): `WARP` 111 k, `NONE` 103 k, `WARP_STAGED` 94 k SPS. O `WARP` é o
  mais rápido aqui (sem DR que reamostre arrays).
- **Compile.** `lower().compile()` 1.1–1.9 s; primeira chamada 0.7–10 s (cresce com `nworld`). Com
  cache de kernels Warp **frio**: 65 s na primeira chamada (7 robôs, 1 024).
- **Iterações do Newton** (`tolerance` 1e-8): média 1.1–1.4 e máx 5 no `random`; média 2.2–3.3 e máx
  13 no `cluster` (com `njmax` estourado o máx vai a 100).
- Escala com `nworld` satura cedo com 7 robôs (1 024): o custo por mundo é o broadphase dos 637 meshes
  que colidem.

## MJX-JAX

- **Compile proporcional a pares de meshes distintos.** O `collision_driver` agrupa pares por
  `FunctionKey(types, data_ids, condim)`: cada par de meshes **distintos** compila uma função. O
  `attach` duplica os meshes por robô, então a variante com uma cópia por robô dava 28 × robôs grupos
  só com as paredes. Apontar todos os robôs para a mesma cópia do mesh fixa em **61 grupos**, sem
  depender do número de robôs. Compile do `step`: ~47 s (2 e 7 robôs); `collision` sozinho ~26 s.
  Isso explica o OOM de compile do 04: o modelo base tem milhares de pares de meshes distintos.
- **Penhasco do Newton.** Com Newton, a 256 mundos ou mais o `step` passa de ~38 ms para **~3.4 s por
  chamada**, constante de 256 a 1 024 mundos e independente de `tolerance`, `iterations` e
  `ls_iterations` (testado 10/10). Com **CG** some (33 ms a 256). Cholesky e `cho_solve` isolados (e
  dentro de `while_loop` com `vmap`) não reproduzem; causa não isolada. Newton abaixo do penhasco:
  3.6 k SPS (2 robôs, 128 mundos); 7 robôs a 128 mundos: 522 SPS, compile 259 s, 9 GB de GPU.
- `solver_niter` volta 0 no MJX-JAX (`_impl.solver_niter`): o número de iterações do Newton/CG por
  passo **não é observável** por esse campo. Proxy: `tolerance` 1e-6 em vez de 1e-8 dá +18 % de SPS
  (3.2 k vs 2.7 k, 64 mundos, Newton).
- **x64** (`JAX_ENABLE_X64`): 277 SPS (7 robôs, 1 024, CG), 6× mais lento que float32, 8.5 GB.
- **Cápsula × cilindro** no dribbler: mesmo SPS (6.29 k vs 6.31 k, 2 robôs, 4 096). Não há ganho de
  desempenho na troca.
- **Prisma de 8 lados** e o resto das variações a 4 096 mundos com 7 robôs falharam no autotune do
  XLA (`All configs failed during profiling`, provável falta de memória); a 1 024 não foram rodados.

## Precisão do contato bola × barra do dribbler

512 posições da bola em volta de `r_col_barDribbler` (r 9 mm × 72 mm), de 2 mm de penetração a
0.5 mm de folga, ângulo na metade voltada para fora, e de 15 % além das pontas. Referência: CPU no
modelo base (cilindro). "Faltas" = contato numa engine e não na outra.

| comparação | região | contatos | faltas | erro de `dist` p99 | erro de posição p99 | erro da normal p99 |
|---|---|---|---|---|---|---|
| MJX-Warp, base (cilindro) | lado / pontas | 355 / 43 | 0 / 0 | 7e-6 mm | 2e-5 mm | 0.01° |
| MJX-JAX, SDF do cilindro | lado | 236 | **119** | 1.5 mm | 8.0 mm | 10.9° |
| MJX-JAX, SDF do cilindro | pontas | 37 | 6 | 0.5 mm | 2.8 mm | 6.5° |
| cápsula (CPU ou MJX-JAX, idênticos) | lado | 336 | 19 | 1.2 mm | 6.0 mm | 16.3° |
| cápsula | pontas | 2 | **41** | 1.7 mm | 6.2 mm | 16.9° |

- O Warp reproduz o cilindro da CPU no float32.
- O SDF do MJX-JAX **perde um terço dos contatos laterais** rasos e erra a posição do ponto em até
  8 mm: não serve para a posse nem para o kick.
- A cápsula erra de forma determinística, igual na CPU e no MJX-JAX: as pontas arredondadas perdem
  quase todo contato além do fim do cilindro.

## Contatos e restrições na CPU (dimensionamento)

3 000 passos por task com o `goto_ball_policy` (dribbler ligado), e 50 episódios de 1 s de `cluster`:

| fonte | `ncon` médio / máx | `nefc` médio / máx |
|---|---|---|
| GoToBall | 1.0 / 13 | 10.6 / 52 |
| GoToBallShoot | 1.1 / 5 | 6.9 / 20 |
| Dribbling | 1.4 / 4 | 11.2 / 22 |
| StaticDefenders | 1.5 / 13 | 11.5 / 46 |
| ContestedPossession | 1.0 / 15 | 8.8 / 60 |
| PassEndurance | 1.2 / 2 | 7.8 / 14 |
| cluster, 2 robôs | 1.5 / 7 | 11.1 / 34 |
| cluster, 7 robôs | 8.1 / 52 | 36.8 / 159 |

`nconmax` na CPU nunca passa de 52, mas no Warp isso não dimensiona nada: vale o broadphase acima.

## Para os tickets seguintes

- **06 (variante e meta de speedup).** O MJX-JAX, mesmo com toda colisão de robô trocada por um
  prisma, fica 40–50× abaixo da CPU com 12 threads e com contato bola × barra pior que o da cápsula
  nas laterais. Nenhuma simplificação razoável fecha um fator de 40. O MJX-Warp roda o modelo base
  inteiro, exato em float32, mas o ganho sobre a CPU é modesto: 3.7× com 2 robôs e 1.4× com 7.
- **09 (paridade).** Warp × CPU no contato da barra: diferença na casa de 1e-5 mm. A diferença de
  motor vai vir da dinâmica (float32, solver), não da geometria.
- **Buffers do Warp**: o spec precisa checar `overflow` (máscara) por passo e dimensionar o
  `naconmax` pelo broadphase, por cena; num lote de 7 robôs, o pool de contatos é o que limita `nworld`.
