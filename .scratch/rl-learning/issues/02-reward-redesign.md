# Redesenho das recompensas das seis tasks

Type: task
Status: implementado, aguardando treino
Blocked by: —

Especificação do usuário (2026-11-03), revisada em `01-why-nothing-learns.md`. Taxa de política
15 Hz (`sac_agent.py`); γ 0.99. Notação: `Δx = x_anterior − x_atual`; `cos(θ)` = cosseno entre a
frente do robô e o vetor robô→bola, **puro** (não é diferença de potencial).

## Comum a todas

| termo | λ | função |
|---|---|---|
| control | 0.01 | `−‖a‖²` (era 0.001) |
| tombo | 1.0 | `−100 · tombou`, terminal, nos dois tiers (no tier 0 é no-op: junta planar) |

## Por task

| env | termos | bônus | terminal |
|---|---|---|---|
| GTB | 1.0·Δ(robô→bola), 0.1·cos θ | +100 ao chegar | chegou, ou 10 s |
| GTBS | 1.0·Δ(robô→bola)·¬has_kicked, 1.0·Δ(bola→gol)·has_kicked, 0.1·cos θ | +100 gol | gol; bola fora (−100); 12 s |
| Drib | 1.0·Δ(robô→bola), 0.1·cos θ, 1.0·held, 1.0·Δ(bola→waypoint atual)·held | +10 por gate (bola a < tolerância do waypoint, com posse) | todos os waypoints; > 2 s sem posse; 30 s |
| SD | as do GTBS + colisão | +100 gol | as do GTBS; colidiu com defensor (−100, qualquer contato); 20 s |
| CP | 1.0·Δ(robô→bola), 0.1·cos θ, 1.0·held, colisão | +100 ao segurar 2 s seguidos | sucesso; colisão de *crashing* (−100); 20 s |
| PE | 1.0·Δ(robô→bola)·¬has_kicked, 1.0·Δ(bola→receptor)·has_kicked, 0.1·cos θ, 0.1·cos β | +100 recepção | recepção; bola fora (−100); 12 s |

- `has_kicked`: o kicker ou o chipper disparou neste episódio.
- Drib: holding domina de propósito (segurar sempre, passando pelos gates). Tolerância do gate =
  `waypoint_tol` atual, 0.3 m.
- PE, β: ângulo entre a frente do chutador e o vetor chutador→receptor (a ideia é "olhar para o
  receptor"; cos entre as duas frentes daria −1 no frente a frente).
- CP, *crashing* (RULEBOOK, SSL rules §8.4.2): no instante da colisão, a diferença das velocidades
  dos dois robôs projetada na linha entre eles; > 1.5 m/s é falta do mais rápido. Penaliza só
  quando o agente é o mais rápido.

## Decisões

- (b) `has_kicked` nunca volta a falso: um chute por episódio (usuário: reset só faria sentido em
  rebote no SD, e o regulamento já pune o *self-pass*).
- `has_kicked` conta só o disparo com o robô na bola (< `REACH`, cos > 0.9, no estado em que a ação
  foi tomada). Desvio do "o kicker disparou" literal: a exploração aleatória dispara em metade dos
  passos e gastaria o chute no primeiro passo, desligando o termo de aproximação.
- CP: a cláusula "< 0.3 m/s, os dois cometem falta" da §8.4.2 ficou de fora.

## Ranking com scripted (tier 0, seeds 0-4, `scratchpad/rank.py`)

| task | política | retorno | descontado | sucesso |
|---|---|---|---|---|
| GTB 60 Hz | go | 120.9 | 78.0 | 1.0 |
| GTB 15 Hz | go / idle / camp / random | 45.0 / 6.9 / −1.1 / −2.9 | 15.4 / 3.6 / 0.2 / −1.4 | 0.4 / 0 / 0 / 0 |
| GTBS 60 Hz | shoot | 127.9 | 43.6 | 1.0 |
| GTBS 15 Hz | shoot / idle / camp / random / aproxima e para | 35.8 / 6.4 / 1.4 / 0.0 / −55.2 | 14.2 / 3.0 / 1.6 / 0.5 / −30.7 | 0.6 / 0 / 0 / 0 / 0 |
| Drib 15 Hz | dribble / **parado com a bola** / idle / random | 200.8 / **490.3** / 4.5 / −4.4 | 86.0 / **107.8** / 3.9 / −3.8 | 0 / 0 / 0 / 0 |

- Os scripted foram sintonizados a 60 Hz; a 15 Hz o P-controller oscila (GTB 0.4). Não é o env.
- Drib: ficar parado com a bola rende mais que carregá-la. Aceito de propósito pelo usuário.
- O dribble scripted perde a bola (giro de até 4 rad/s, acima do limite de ~2 rad/s do ticket 22).

## Drib: o sucesso paga o holding restante (opção A)

No sucesso, `+1.0 × (max_steps − steps)`: completar nunca vale menos que ficar parado com a bola.

## Drib: retenção andando e girando (tier 0, seeds 0-2, rampa suave)

Andar reto até 0.6 m/s segura 5 s; **qualquer giro andando perde a bola em ~1 s**, já a 0.3 m/s e
0.5 rad/s. No frame do robô a bola rola de lado ao longo do rolete (y: 0 → −24 mm) até escapar.

Mecanismo (Coulomb, sem trapaça): o rolete a 314 rad/s escorrega na bola a ~2.8 m/s; o atrito
cinético aponta na direção do escorregamento total, então a componente lateral é ~μN·v_lat/v_slip,
~1 % de μN. Quanto mais devagar o rolete, mais aderência lateral:

| rolete | 0.3 m/s + 0.5 rad/s (passos segurados de 75, 5 s a 15 Hz) | giro no lugar 1 rad/s |
|---|---|---|
| 314 rad/s | 15 | 9 |
| 150 | 49 | 15 |
| 80 | **73** | 9 |
| 40 | 73 | 9 |

Scripted dribble (15 Hz, giro máx. 1.5 rad/s, seeds 0-4) com a opção A:

| rolete | retorno | waypoints | sucesso |
|---|---|---|---|
| 314 rad/s | 356.5 | 1 1 2 1 2 | 0 |
| **78.5 rad/s** | **526.1** (> 490 parado) | 1 2 2 4 3 | 0.2 |

A velocidade do rolete já está na ação, então o agente pode aprender a usá-la. Giro no lugar a
1 rad/s ainda perde a bola nesta medição (rampa curta de 8 passos), o que contradiz o ~2 rad/s do
ticket `physics-fidelity/22`: aberto.

## Plano

1. `tasks.py` por env; `horizon_s` por task; tombo e control no `BaseEnv`/`_BallTask`.
2. Testes de ranking com os scripted: completar > acampar > parado > aleatório.
3. Docs de cada env.
4. Treino 1M passos tier 0, GTB primeiro.
