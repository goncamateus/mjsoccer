# Kick no MJX-Warp: substeps, `timestep` por mundo e a tabela (ticket 15)

Medido em 2026-11-09 no `~/.venvs/mjsoccer-gpu` (`mujoco`/`mujoco-mjx[warp]` 4.0.0, RTX 3060).
Script `15-kick/kick15.py`, saída bruta `15-kick/results.jsonl` (`results_3dp.jsonl` = primeira
rodada a 3 casas), log `15-kick/log.txt`.

## Montagem

- Partida: `Dribbling-v0` tier 0, `reset(seed=0)` + 30 passos de ação zero (o assentamento do
  `calibrate.py`); snapshot do `qpos/qvel/act`. Robô 0 dispara kicker ou chipper com força
  `F ∈ {0, 25, …, 500}` N durante o passo de controle de disparo; `ctrl` zero no resto.
- Passo de controle = 24 passos de física (40 Hz pós-port). Velocidade de saída = pico da velocidade
  3-D da bola nas fronteiras de passo de controle, por 30 passos.
- **Esquema K**: no passo de disparo, K passos a `dt/4` e `24 − K/4` a `dt`. K = 96 é o mecanismo de
  hoje (passo inteiro a `dt/4`); K = 0 é sem substep. **Curto**: 24 passos a `dt/4` e nada mais (o
  passo de disparo dura 6.25 ms em vez de 25 ms).
- No Warp, `opt.timestep` por mundo via `vmap(mjx.step, in_axes=(in_ax, 0))` com só
  `opt.timestep` em eixo 0; mundos que não disparam andam 24 passos a `dt` e ficam congelados no resto
  das `L = max(K + 24 − K/4, 24)` iterações (por `Data.where` ou por `timestep = 0`).
- CPU = `mujoco 4.0.0` float64 com `mj_step` cru sobre o mesmo snapshot.

## 1. Paridade Warp × CPU

Diferença relativa máxima sobre os 20 pontos de força × 2 plungers:

| esquema | kick | chip |
|---|---|---|
| K = 0 | 0.0002 % | 0.0006 % |
| K = 32 | 0 (a 3 casas) | 0 (a 3 casas) |
| K = 96 | 0.0002 % | 0.0001 % |
| curto | < 0.0001 % | < 0.0001 % |

O Warp em float32 reproduz o CPU até a precisão do float32, inclusive o serrilhado de K = 0. Os
dois mundos que não disparam ficam parados (0.0004 m/s), `time` igual em todos os mundos (espalhamento
0), `overflow` 0. **Não precisa de tabela float32 própria**: a mesma tabela serve aos dois backends.

## 2. O que o substep compra (CPU, m/s)

Kick, pontos 25–300 N (o teto de 6.5 m/s cai em ~285–365 N):

| K | 25 | 50 | 125 | 150 | 200 | 225 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.631 | 2.049 | 4.283 | 4.696 | **5.636** | **5.425** | 6.107 | **6.307** (< 6.789 em 275 N) |
| 8 | 0.640 | 2.065 | 4.294 | 4.705 | **5.635** | **5.421** | 6.095 | 6.290 |
| 16 | 0.647 | 2.074 | 4.282 | 4.677 | 5.594 | 5.560 | 6.035 | 7.413 |
| 32 | 0.651 | 2.068 | 4.246 | 4.427 | 5.090 | 5.529 | 5.675 | 6.388 |
| 48 | 0.618 | 2.052 | 3.876 | 4.346 | 5.009 | 5.475 | 5.640 | 6.382 |
| 96 | 0.584 | 1.993 | 3.896 | 4.366 | 5.033 | 5.500 | 5.668 | 6.382 |
| curto | 0.0006 | 0.006 | 2.300 | 3.081 | 4.517 | 5.252 | 5.650 | 6.270 |

- K ≤ 16 serrilha (curva não monótona dentro da faixa útil): o substep é necessário, como no ticket 09
  do `physics-fidelity`.
- K = 32 é o menor esquema monótono nos dois plungers; o chip já é idêntico ao K = 96, o kick fica até
  9 % acima em 125 N. K ≥ 48 fica a ≤ 1 % do K = 96 exceto em 25 N.
- O curto é monótono até o teto, com zona morta abaixo de ~75 N (kick) — curva própria, tabela própria.
- **Tabela atual × CPU 4.0 a 40 Hz (K = 96)**: kick 1–3 % abaixo da `KICK_TABLE_MS[0, 0]`
  (125 N 3.896 × 3.93; 250 N 5.668 × 5.79; 225 N 5.50 × 5.62). A tabela foi medida no 3.11 a 60 Hz pelo
  `env.step`: a recalibração entra no upgrade para o 4.0 (restrição 12), seja qual for o mecanismo.

## 3. Custo (2 robôs, 4 096 mundos, física crua, 1 em 20 mundos disparando todo passo)

| mecanismo | iterações | decisões/s | × base |
|---|---|---|---|
| base: `timestep` único, sem disparo | 24 | 35.0 k | 1.0 |
| `timestep` por mundo (`in_axes`), sem máscara | 24 | 35.2 k | 1.0 |
| `Data.where` por passo, `timestep` único | 24 | 20.8 k | 0.59 |
| `timestep` por mundo + `Data.where`, K = 0 | 24 | 16.2 k | 0.46 |
| K = 96 + `Data.where` | 96 | 5.0 k | 0.14 |
| **K = 96, ociosos congelados por `timestep = 0`** | 96 | 9.2 k | 0.26 |
| lote todo a `dt/4` | 96 | 9.2 k | 0.26 |
| K = 32, `timestep = 0` | 48 | 17.8 k | 0.51 |
| curto, `timestep` por mundo | 24 | 34.4 k | 0.98 |

- O `timestep` por mundo sai de graça; o caro é o `Data.where` por passo (copia o `Data` inteiro):
  −40 %. Congelar com `timestep = 0` é exato (paridade e `time` idênticos, sem NaN) e só custa as
  iterações.
- O custo escala com as iterações do passo mais longo do lote. Com 4 096 mundos e cooldown de
  10 decisões, quase todo passo tem algum mundo disparando: um `lax.cond` no nível do lote
  ("só substepa se alguém disparou") não ajuda. O mecanismo de hoje custa ~3.8× sempre.
- 8 192 mundos: a base estourou a GPU neste script (alocação do Warp com o JAX a 0.9); K = 96 rodou
  a 9.6 k. Não perseguido — o default de 2 robôs com `WARP` é 8 192 no treino (ticket 14) e vale
  checar lá.
