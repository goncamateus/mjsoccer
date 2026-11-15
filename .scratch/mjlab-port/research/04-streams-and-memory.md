# 04 — Streams torch × Warp e memória do device (ticket 05)

Medido em 2026-11-14, RTX 3060 12 GB, mujoco/mujoco-warp 4.0.1, warp-lang 1.18.0, torch
2.14.1+cu130, `ContestedPossession-v0` (2 robôs, `nconmax` 64, `njmax` 256), sob
`systemd-run --user --scope -p MemoryMax=10G`. Scripts em [`../bench/`](../bench/).

## Resumo

- **Memória**: a diferença entre o device (8,9 GB a 16 384 mundos) e o mempool do Warp
  (1,3 GB) era **memória de grafo CUDA**: 7,2 GB de scratch do EPA (colisão convexa), que o
  Warp dimensiona por `naccdmax` e cujo default é `naconmax` (64 slots por mundo). O pico
  medido com os robôs se empurrando em todos os mundos é 0,08 slot por mundo. Com
  `naccdmax = 1 × nrobôs × nworld` (`scene.NACCD_PER_ROBOT`), 16 384 mundos usam **2,0 GB**
  e o maior `nworld` que roda vai de 16 384 para **98 304** (9,8 GB; o maior testado), com o
  mesmo dec/s.
- **Streams**: o stream do Warp é bloqueante, então o torch no stream default (nulo legado)
  fica ordenado com ele sem nada explícito: 0 corrupções. Num side stream do torch sem o
  nosso wrapper, **todo** mundo corrompe. O wrapper (torch no stream do Warp dentro de
  `step`/`reset`/`refill_pool`) fica: o dec/s fica dentro do ruído e ele protege qualquer
  caller. Teste de
  regressão em `tests/test_mjlab_smoke.py::test_a_caller_on_a_side_stream_does_not_race_warp`.

## 1. Memória decomposta por fonte

[`mem_mjlab.py`](../bench/mem_mjlab.py): um `make` por processo, atributos do driver
(`cuMemGetInfo`, `cuMemPoolGetAttribute` no mempool default do device, que é o do Warp,
`cuDeviceGetGraphMemAttribute`) e o caching allocator do torch. MB, após 50 decisões.
"Antes" rodou no HEAD do ticket 04 (`naccdmax` default do Warp), "depois" com este ticket →
[`mem_mjlab.txt`](../bench/mem_mjlab.txt). A base de 8,9 GB (e não os 10,9 GB do spec) é a
do ticket 04, que já tinha baixado de 10,9 para 9,0 GB ([`03-env-dps.md`](03-env-dps.md)).

| fonte | 1 024 antes | 4 096 antes | 16 384 antes | 16 384 depois |
|---|---:|---:|---:|---:|
| outros processos (desktop) + contexto CUDA torch/Warp | 369 | 364 | 364 | 383 |
| mempool do Warp, reservado (usado) | 128 (99) | 352 (339) | 1 312 (1 298) | 1 312 (1 298) |
| **memória de grafo CUDA** | **448** | **1 824** | **7 168** | **256** |
| torch, reservado (alocado) | 24 (9) | 28 (12) | 74 (25) | 74 (25) |
| resto (código de módulo, workspaces) | 29 | 34 | 34 | 34 |
| **device usado** | 998 | 2 602 | **8 952** | **2 059** |

O device tinha 234 MB em uso antes de qualquer processo (desktop); o contexto CUDA de torch +
Warp custa ~130-150 MB, uma vez só (os dois usam o contexto primário).

Os candidatos do protótipo:

- **Mempool reservado**: não. Reservado ≈ usado (1 312 contra 1 298 MB).
- **Contexto CUDA torch + Warp**: não. ~140 MB, fixo.
- **Grafo capturado duas vezes** (no `__init__` do mjlab e no nosso `create_graph()`): não
  como memória residente. A memória de um grafo só é mapeada no primeiro launch, e os
  grafos do `__init__` nunca rodam (são substituídos antes). Mas a captura do `forward` no
  `__init__` aloca na hora com o `naccdmax` default, e é **ela** que falhava a 32 768 mundos
  (`Failed to allocate 1518338048 bytes`, em `collision_convex.py:1322`) mesmo depois de
  resolver o grafo. Por isso o `naccdmax` entra já no `put_data` (`env._naccdmax`), não só
  antes da nossa recaptura.
- **O que era**: `wp.empty` dentro de `mjwarp.step`/`forward`, capturados no grafo
  (alocações de grafo, fora do mempool que o Warp reporta). Rastreados por linha
  (`array._init_new` instrumentado, 1 024 mundos): 452 MB, **todos** em
  `collision_convex.convex_narrowphase` — `epa_pr` 135, `epa_vert` 60, `epa_face` 45,
  `epa_norm2` 45, `multiccd_*` ~150, `epa_horizon` 6. Cada um tem `d.naccdmax` linhas;
  ~7 KB por slot (`ccd_iterations` 35 da cena, multiccd ligado por haver cilindros).
  `step` e `forward` reaproveitam a mesma memória física (448 MB os dois juntos).

### Quantos slots EPA a cena precisa

Um slot é tomado só quando o GJK acha um par convexo **penetrando**
(`collision_convex.py:862`), contado no lote inteiro, por tipo de par.
[`epa_slots.py`](../bench/epa_slots.py), eager (sem grafo), contador lido por passo →
[`epa_slots.txt`](../bench/epa_slots.txt):

| cenário | mundos × decisões | pico de slots no lote | por mundo |
|---|---:|---:|---:|
| ações uniformes (o bench) | 1 024 × 60 | 0 | 0 |
| azul mirando o amarelo em todo mundo (choque físico) | 512 × 120 | 39 | 0,076 |
| azul teleportado encavalado no amarelo (0,12–0,20 m) | 256 × 1 | 4 442 | 17,4 |

O caso teleportado já estoura BROADPHASE com `nconmax` 64 (todos os 256 mundos terminaram);
não é física alcançável. `NACCD_PER_ROBOT = 1` dá 2 slots por mundo na cena de 2 robôs, 26×
o pico com todo mundo em choque. O pico vem de um cenário (512 × 120), e a bola presa no
drible entra nele e nas ações uniformes (o pool assenta a bola no drible; `dribble > 0` em
metade das ações) com 0 slots: esfera × malha só pede EPA quando penetra. O estouro é como o do BROADPHASE (buffer único do lote): o
bit `OverflowType.CCD` cai nos mundos que chegaram por último, que terminam zerados pelo
`settle` (qualquer bit termina). A cena de 7 robôs fica com 7 slots por mundo; medir no
ticket da GoToBall.

## 2. `nworld` máximo, remedido

[`dps_mjlab.py`](../bench/dps_mjlab.py), um processo por tamanho, melhor de 3 × 10 s →
[`results_mjlab_05.txt`](../bench/results_mjlab_05.txt). Antes = ticket 04
([`03-env-dps.md`](03-env-dps.md)).

| nworld | dec/s | device antes | device depois | mempool do Warp (pico) |
|---:|---:|---:|---:|---:|
| 4 096 | 16 531 | 2 600 MB | 842 MB | 339 MB |
| 16 384 | 17 492 | 8 952 MB | 2 040 MB | 1 298 MB |
| 32 768 | 17 564 | OOM | 3 602 MB | 2 576 MB |
| 65 536 | 16 954 | — | 6 686 MB | 5 132 MB |
| 98 304 | 16 488 | — | 9 827 MB | 7 688 MB |

0 overflow em todos. dec/s igual ao ticket 04 onde comparável (16 598 / 17 534). O dec/s
satura em ~17,5 k a 16-32 k mundos e cai um pouco acima. 131 072 projeta ~13 GB e não
cabe nos 12 GB; entre 98 304 e 131 072 não foi testado. "Roda" aqui é ~40 s de passos, não
um treino longo (ticket 11).

## 3. Streams

### O que o código faz

- O Warp cria seu stream com `CU_STREAM_DEFAULT` (`warp.cu`, `wp_cuda_stream_create`), ou
  seja **bloqueante**: ele sincroniza implicitamente com o stream nulo legado, que é o
  default do torch (`torch.cuda.current_stream().cuda_stream == 0`).
- O mjlab lança os grafos no stream do Warp, e o `TorchArray.__setitem__` escreve no stream
  do Warp. Leituras (`d.qpos[:]`) e todo o resto do torch rodam no stream corrente do
  torch. O mjlab não sincroniza nada.
- O `SoccerEnv` roda `step`/`reset`/`refill_pool` dentro de `torch.cuda.stream(warp)`, com
  `wait_stream` na entrada e na saída.

### Medição

A física do Warp **não é bit-reprodutível** entre execuções: dois `make` com a mesma seed
já têm pools diferentes (até 1,1 mm em `qpos`, todo mundo), por atomics no contato e no
solver. Comparar execuções não serve. [`streams_mjlab.py`](../bench/streams_mjlab.py)
checa um invariante dentro de cada execução: o torch escreve o `opt.timestep` por mundo em
cada iteração (`SoccerEnv.substep`) e o grafo soma isso em `data.time`, então o `time` de
cada mundo tem de ser **exatamente** a soma em float32 do plano que o torch escreveu.
4 096 mundos × 200 decisões (mundos que terminaram ficam de fora) →
[`streams_mjlab.txt`](../bench/streams_mjlab.txt):

| modo | dec/s | mundos checados | `time` errado | erro máximo |
|---|---:|---:|---:|---:|
| `warp` (como entregue) | 16 407 | 4 093 | 0 | 0 |
| `default`: sem o wrapper, torch no stream default | 15 934 | 4 092 | 0 | 0 |
| `side`: sem o wrapper, caller num `torch.cuda.Stream()` | 15 408 | 1 313 | **1 313** | 2,77 s |

No `side`, 2/3 dos mundos terminaram (lixo) e todo o resto tem o relógio errado.

Limites do detector: ele só vê uma corrida na direção escrita do torch → leitura do Warp, e
só nas iterações em que o timestep muda (24 → 0 em todo mundo ocioso, dt/4 → dt e dt → 0
num que disparou; toda decisão tem ao menos uma). Uma leitura atrasada de `qpos` pelo torch
não aparece nele. O `default` limpo é coerente com a semântica do CUDA (stream bloqueante ×
stream nulo legado), que é o argumento principal; a medição só não o contradiz. O `side`,
ao contrário, prova a corrupção. Uma execução por modo: os ±3 % de dec/s entre modos são
ruído, não custo.

### Decisão

**Manter o torch no stream do Warp** (o wrapper). Sem ele, o caso comum (RSL-RL no stream
default) seria seguro só por causa de um detalhe de implementação do Warp (stream
bloqueante) e o caso de um caller em side stream corromperia em silêncio; com ele, os dois
são seguros e o custo fica no ruído (o modo `warp` foi o mais rápido). Fica documentado no
docstring de `mjsoccer/mjlab/env.py`, e o teste do smoke falha sem o wrapper (243 de 243
mundos com `time` errado, verificado à mão trocando o wrapper por `nullcontext` no `step`).

## Fora do escopo, para depois

- **O MJX-Warp tem o mesmo default**: `mjx.make_data(..., naconmax=...)` em
  `mjsoccer/mjx/base.py` não passa `naccdmax`. É provável que o OOM do bench MJX a 8 192
  mundos seja isto. Mexer exige reabrir a paridade e o `ppo_mjx`; vale um ticket.
- O pool de primeiros estados escala linear (44 s a 98 304 mundos); não importa para o
  default de `--num-envs`, que o ticket 11 escolhe.
