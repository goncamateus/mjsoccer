# Por que nenhum modelo treinado aprende

Type: task
Status: in progress
Blocked by: —

## Pergunta

O sweep SAC (`runs/sweep/sac_20261018-231040`, 1M passos, 6 envs × 2 tiers, `examples/sac_agent.py`)
não aprendeu nada além de GoToBall parcial (sucesso 0.40 tier 0, 0.18 tier 1; o resto 0-0.01).
Hiperparâmetro ou simulador?

## Método

Separar as três causas por task com controladores scripted (sem treino, segundos):
- **simulador**: um controlador scripted resolve a task?
- **reward**: ele rende mais (descontado e não descontado) que políticas degeneradas (parado,
  girar no lugar, aleatória)?
- **hiperparâmetro**: só depois dos dois acima, por treinos diferenciais (uma variável por vez).

## Achados

| task | simulador (scripted) | reward | gargalo |
|---|---|---|---|
| GoToBall | 100 % nos dois tiers | `0.05 * facing` por passo: girar para a bola e parar rendia 59.5 vs 29.7 de chegar | reward + horizonte |
| GoToBallShoot | 100 % nos dois tiers (o chute funciona) | sem termo por passo; gol vale 0.66 descontado (γ 0.99) vs 0 parado; energia aleatória −3.6/episódio | horizonte + exploração |
| Dribbling | perde a bola girando ≥ 1 rad/s | — | simulador (ticket `physics-fidelity/22`) |
| Contested / Pass / StaticDefenders | não testado | — | — |

**Horizonte.** A 60 Hz, γ 0.99 enxerga ~1.7 s; as tasks levam 5-8 s (o scripted chega na bola em
~300 passos, faz gol em ~450). Repetir a ação 4 passos (frame-skip 4, `gradient_steps` 4 para
manter ~62k updates) leva o horizonte a ~6.7 s. `BaseEnv` já aceita `control_hz`, que faz o mesmo
sem wrapper.

**Updates.** 16 envs × `gradient_steps=1` = 1 update por 16 transições (62k em 1M). `ent_coef`
colapsa para 1e-5–1e-4 em todos os runs.

## Treinos diferenciais (tier 0, seed 0, 1M passos de env salvo indicação)

| task | config | avaliação 20 ep. | sucesso no rollout |
|---|---|---|---|
| Shoot | base | 0 | 0.01 |
| Shoot | frame-skip 4 | 0 | 0 (ep_rew −3.7 → −0.33: só aprende a não gastar energia) |
| Shoot | γ 0.999 (300k) | 0 | 0 |
| GoToBall | reward antigo, base (sweep) | — | 0.40-0.44 |
| GoToBall | reward novo, base | 0.15 | 0.28 |
| GoToBall | **reward novo, frame-skip 4**, seeds 0-4 | **0.95 / 1.00 / 0.90 / 0.90 / 0.95** | **0.91-0.98**, episódio 250-346 passos (scripted 285-307) |
| GoToBall | reward antigo, frame-skip 4 | 0.00 | 0.05, episódio 1160-1200 (acampa) |

Os dois fixes são necessários: o horizonte maior sozinho **piora** o reward antigo, porque com γ
efetivo mais longo acampar vale mais que chegar (a γ 0.999 por passo de 60 Hz: 34.7 vs 26.1).

## Fixes

- `e0e59c5` — GoToBall: facing vira potencial, `0.5 * (facing − prev_facing)`; girar no lugar
  rende 0.27. `test_goto_ball_does_not_pay_camping`.
- `9f7d913` — ticket `physics-fidelity/22`: contato bola↔rolete de PU 65A; limite de giro com
  bola ~1 → ~2 rad/s.

## Treinos com `examples/sac_agent.py` a 15 Hz (`c9a29db`, tier 0, seed 0, 250k passos de política)

| task | sucesso no rollout (fim) | ep_rew | episódio | leitura |
|---|---|---|---|---|
| GoToBall | **0.93** | 14.5 | 68 passos (~4.5 s) | caminho nativo `control_hz=15` reproduz o wrapper |
| GoToBallShoot | 0.00 → 0.06 (sobe nos últimos 50k) | 1.75 → 5.2 | ~380 de 450 | aprendeu ir até a bola (scripted "aproxima e para" = 5.5); chute ainda raro |
| Dribbling | 0 | −0.05 → 0.5-0.75 (subindo) | 24 → 48 (~3 s) | segura a bola mais tempo, longe dos waypoints |

Nenhum dos dois platôs parece convergido: as curvas de Shoot e Dribbling ainda sobem em 250k.

## Treinos de 1M passos de política (15 Hz, tier 0, seed 0; avaliação determinística, 20 ep., seeds de reset 0-4 × 4)

| task | 250k | 1M | como terminam os episódios (1M) |
|---|---|---|---|
| GoToBallShoot | sucesso 0.00, ret 3.6 | **sucesso 0.05**, ret 6.7, 282 passos | 14 bola fora, 5 timeout, 1 gol |
| Dribbling | sucesso 0.00, ret 1.0, 50 passos, waypoints 0-1 | sucesso 0.00, **ret 7.3, 97 passos**, waypoints 0:9 1:7 2:3 3:1 (de 4) | 20/20 perdem a bola (> 2 s sem posse) |

**Shoot.** A política empurra a bola para +x (o gol), quase sempre devagar (0.5-3 m/s, conduzindo e não
chutando), mas sem mirar: 7 saem pela linha de fundo fora do gol, 7 pela lateral. Bola fora não tem
penalidade, e o shaping `prev_goal − goal` já pagou o avanço antes da saída. O gradiente está certo
(levar a bola ao gol); o que falta é precisão.

**Dribbling.** Dobrou o tempo de posse e chega a até 3 de 4 waypoints, mas sempre termina perdendo a
bola. Coerente com o limite de giro com bola de ~2 rad/s (ticket `physics-fidelity/22`).

## Pendente

- Shoot: decidir se bola fora deve custar algo (hoje é saída grátis depois de colher o shaping).
- Dribbling: limite de giro de 3 rad/s com bola continua aberto (ticket `physics-fidelity/22`).
