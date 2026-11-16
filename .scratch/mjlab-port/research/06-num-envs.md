# 06 — `--num-envs` default por cena, em treino longo (ticket 11)

Medido em 2026-11-15/16, RTX 3060 12 GB, `examples/ppo_mjlab.py` com os defaults de PPO, um
processo por tamanho, sob `systemd-run --user --scope -p MemoryMax=10G`
([`../bench/train11.sh`](../bench/train11.sh) → [`../bench/train11/`](../bench/train11/)).
Cada época: `refill_pool` + iterações de PPO + eval em 128 mundos + checkpoint e ONNX. O
script agora imprime por época os mundo-decisões de **treino** com cada falha
(diverged/overflow/reset_overlap, contados no device) e a memória do device em uso.

## Resultado

ContestedPossession (2 robôs), 10 épocas, ~31-33 M decisões (~40 min cada):

| nworld | dps | falhas no treino (div/ovf/overlap) | device (min-máx entre épocas) | eval return final |
|---:|---:|---:|---:|---:|
| **8 192** | 14,6k | 0 / 13 / 0 | 1 283-1 354 MB | 0,26 |
| 16 384 | 15,5k | 1 / 3 / 0 | 2 263-2 289 MB | 0,25 |
| 32 768 | 16,1k | 0 / 0 / 0 | 4 053-4 089 MB | 0,11 |

GoToBall (7 robôs), 8 épocas, ~4,6-5,2 M decisões (~50-55 min cada):

| nworld | dps | falhas no treino | device | eval return final |
|---:|---:|---:|---:|---:|
| 4 096 | 2,07k | 0 / 0 / 0 | 1 799-1 833 MB | 0,02 |
| 8 192 | 2,03k | 0 / 0 / 0 | 3 077-3 131 MB | 0,06 |
| **16 384** | 2,05k | 0 / 0 / 0 | 5 601-5 621 MB | 0,09 |
| 32 768 | 2,06k | 0 / 0 / 0 | 10 601-10 657 MB | 0,17 |

Nenhuma época de eval teve falha. A memória é plana em todos (a variação, ≤ 70 MB, não cresce
com as épocas): sem vazamento por refill, eval ou export.

## Decisão

- **2 robôs: 8 192** (mantido). O dps satura (+10 % de 8 192 a 32 768) e, no mesmo orçamento,
  32 768 faz 4× menos atualizações de PPO e aprende menos (0,11 contra 0,26).
- **7 robôs: 16 384** (era 2 048, o do MJX). Estável e com folga; 32 768 também roda, mas usa
  10,6 dos 12 GB (quebra com o desktop ou outro processo na GPU). dps igual em todos.

## Achado: overflow raro no treino da ContestedPossession

13 mundo-decisões com overflow em 31 M (8 192) e 3 + 1 diverged em 33 M (16 384), ~10⁻⁷;
nenhum em 32 768, cuja política aprendeu menos, e nenhum na GoToBall. Não cresce com o
`nworld`: depende do que a política faz (o protótipo, de ~40 s com ações uniformes, não via).
O mundo afetado termina com obs e recompensa 0 (`TaskLogic.settle`) e os outros seguem; o
treino não aborta. O tipo do bit não está medido: o contador do treino soma só a métrica, e o
`--eval` com a política de 8 192 em 16 384 mundos não terminou em 20 min (ONNX na CPU). Para
achar: guardar `env.sim.data.overflow` OR-ado por tipo no wrapper do `step` do script.
