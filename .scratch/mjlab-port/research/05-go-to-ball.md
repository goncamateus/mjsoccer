# GoToBall no mjlab (7 robôs): ITERATIONS, buffers, `nworld`, decisões/s

Ticket 08. RTX 3060 12 GB, mujoco 4.0.1, mjlab `6ab9846`. Bench:
`.scratch/mjlab-port/bench/dps_mjlab.py` (ganhou contagem de bits por tipo, picos de
`ncollision`/`nefc`/`solver_niter` lidos no fim de cada decisão antes do auto-reset, e
`CHASE=1`). Saída crua:

- `bench/results_mjlab_08_iso.txt`: **um processo por `nworld`**, a fonte de memória e de
  `nworld` máximo abaixo; mais ContestedPossession a 4 096 para os buffers de 2 robôs.
- `bench/results_mjlab_08.txt`: a primeira passada, vários `nworld` no mesmo processo. Vale
  para decisões/s e bits; **não** para memória (os envs anteriores continuam vivos e o
  high-water do mempool é do processo) nem para o teto (32 768 deu OOM ali e roda isolado).

MJX-Warp: `.scratch/gpu-port-impl/10-bench/results.txt`.

## #1263 (ITERATIONS / LS_ITERATIONS): não aparece

Nenhum bit de overflow, de nenhum tipo, em ~4 M decisões-mundo somando as duas passadas
(256 a 32 768 mundos com ação aleatória, mais 1 024 e 4 096 com `CHASE=1`, o robô 0
atravessando os 6 parados); `rate_ITERATIONS = rate_LS_ITERATIONS = 0.0` explícitos em
cada linha da passada isolada. O bit é pegajoso até o reset do mundo e o bench o lê antes do
auto-reset, então uma iteração de solver estourada em qualquer das 48 de uma decisão
apareceria. No lote default do treino (`NUM_ENVS[7] = 2048`): taxa 0 em 113 k
decisões-mundo (55 por mundo).

Indício a mais, mais fraco: no último substep de cada decisão (o único amostrado) o Newton
convergiu em no máximo 4 iterações de 100 (5 na ContestedPossession). Os avisos do #1263 são
do Go1 do próprio mjlab (pés articulados, outro regime de contato).

## Buffers por mundo

Dimensionados como no MJX (`scene.NACON_PER_ROBOT` × robôs; `njmax = 4 × nconmax`), sem
mudança de código: `make_cfg` já fazia isso para qualquer número de robôs. Medido no mjlab,
ação aleatória (GoToBall: pior caso entre todas as linhas, CHASE incluso):

| | ContestedPossession (2 robôs) | GoToBall (7 robôs) |
|---|---|---|
| `nconmax` / pico de candidatos do broadphase por mundo | 64 / 40,0 | 168 / 120,8 |
| `njmax` / pico de `nefc` (o pior mundo) | 256 / 15 | 672 / 67 |

O pico de candidatos é o mesmo do MJX (120,5) e é o que manda: o broadphase de 7 robôs
parados a 0,5 m já dá ~120 por mundo antes de qualquer contato. Folga de 39 % no
`nconmax`. O `njmax` tem folga de 10× (o mesmo excesso do MJX); encolher a regra mexe nos
dois backends e fica para quem precisar de mais mundos (ticket 11).

## `nworld` máximo

Um processo por linha:

| nworld | dec/s | decisões por mundo | mempool Warp (pico) | device usado |
|---|---|---|---|---|
| 2 048 | 2 297 | 55 | 0,5 GB | 1,1 GB |
| 8 192 | 2 072 | 29 | 2,0 GB | 2,9 GB |
| 16 384 | 2 049 | 26 | 3,9 GB | 5,2 GB |
| 24 576 | 2 030 | 23 | 5,8 GB | 7,5 GB |
| 32 768 | 1 997 | 23 | 7,6 GB | 9,9 GB |
| 40 960 | OOM na criação do grafo CUDA | | | |

**Maior `nworld` que roda: 32 768** (0 overflow, 0 NaN). "Estável" aqui quer dizer 23
decisões por mundo (20 de aquecimento + 3 cronometradas de ~10 s cada), não um treino: o
run longo é o ticket 11. 40 960 não cabe nos 12 GB. Memória por mundo ~0,23 MB de mempool,
~3× a da cena de 2 robôs (0,08 MB a 16 384 no ticket 05). O throughput satura a partir de
512 mundos e cai ~13 % de 2 048 a 32 768, então mais mundos não trazem decisões/s.

## Decisões/s ponta a ponta contra o MJX-Warp

Primeira passada (mesmo processo; decisões/s não depende disso: 2 048 isolado deu 2 297):

| nworld | mjlab | MJX-Warp | Δ |
|---|---|---|---|
| 256 | 1 990 | 2 029 | −1,9 % |
| 512 | 2 183 | 2 185 | −0,1 % |
| 1 024 | 2 194 | 2 211 | −0,8 % |
| 2 048 | 2 189 | 2 206 | −0,8 % |
| 4 096 | 2 108 | 2 131 | −1,1 % |
| 1 024, CHASE | 2 160 | 2 273 | −5,0 % |

Empate dentro de ~1-2 % com ação aleatória, como na ContestedPossession; o bench do MJX
para em 4 096 (compilação de 40 s), o mjlab vai a 32 768. O gap de 5 % com `CHASE` não foi
investigado; hipótese não medida: o CHASE do mjlab é torch fora do grafo a cada decisão, o
do MJX está dentro do `fori_loop` jitado. A GoToBall é ~7-8× mais lenta que a
ContestedPossession nos dois backends (7 robôs, ~3× os candidatos de contato).

## Paridade

- Task-logic parity (`tests/test_mjlab_task_logic.py`): GoToBall nas seeds 0-4, `place` com
  candidatos injetados incluindo o caminho de rejeição do `_scatter`, overlap por mundo.
- `test_parity_seeded` GoToBall × mjlab: `random`/`competent` × `ret`/`held_frac`, as 4
  passam. Os cenários scriptados da GoToBall (`spin`, `envelope`, `bounce`, `navigate`)
  agora rodam na própria cena, não mais transplantados para a da ContestedPossession:
  `-m validation -k mjlab` dá 90 pass, 20 skip (as 4 tasks não portadas).
