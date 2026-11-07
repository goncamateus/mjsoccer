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

## GoToBall depois do redesenho (`ee4aec1`): regressão de reward, não de hiperparâmetro

Sintoma: o sweep de 1M passos com o reward do issue 02 não aprendia GoToBall (logs perdidos).
Loop: `sac_agent.py --env GoToBall-v0 --tier 0 --timesteps 250000`, mesmos hiperparâmetros; sucesso
no rollout ao fim. Uma mudança do `ee4aec1` revertida por vez sobre o HEAD:

| variante | seeds | sucesso | ep_len |
|---|---|---|---|
| HEAD (`ee4aec1`) | 0-4 | 0.12 / 0.13 / 0.20 / 0.10 / 0.12 | ~140 de 150 |
| `e0ff917` (reward antigo) | 0 | 0.93 | 68 |
| horizonte 10 → 20 s | 0-1 | 0.31 / 0.31 | |
| energia 0.01 → 0.001 | 0-1 | 0.13 / 0.23 | |
| bônus 100 → 10 | 0-1 | 0.45 / 0.66 | |
| facing 0.1 → 0 | 0-1 | 0.07 / 0.00 | |
| reward inteiro × 0.1 | 0-1 | 0.12 / 0.20 | |
| × 0.1 e horizonte 20 s | 0-1 | 0.31 / 0.25 | |
| bônus 10 e horizonte 20 s | 0-1 | 0.78 / 0.75 | ~160 |
| **bônus 10, facing potencial `0.5·Δcos`, 10 s** | 0-2 | **0.97 / 0.95 / 0.95** | 61-77 |
| bônus 10, facing potencial, 20 s | 0-2 | 0.95 / 0.98 / 0.93 (0.9 já em ~140k) | 61-66 |

- Escala absoluta descartada: × 0.1 no reward inteiro = HEAD. Não é `ent_coef`, `VecNormalize` nem a
  biblioteca (o `ent_coef` do run antigo é menor ainda, 0.0013).
- Causa: a proporção bônus/shaping. O facing cru por passo exigia bônus 100 para que estacionar não
  pagasse; contra 100 o shaping denso some. Facing potencial não paga estacionado, então o bônus
  volta a 10 e o shaping volta a guiar.
- Fix (opção 2): só GoToBall, `GOTO_BONUS = 10` e facing potencial; horizonte segue 10 s. Verificação,
  seeds 0-4: **0.97 / 0.95 / 0.95 / 0.92 / 0.96**, ep_len 58-77.
- Seam: o loop de aprendizado (~35 min) não vira teste. O que dá para travar é a propriedade:
  `test_goto_ball_does_not_pay_camping` volta a exigir que girar no lugar renda < 1 (no `ee4aec1`
  rendia 58.7 a 60 Hz). O ranking com scripted passava no reward quebrado: ele não vê a proporção.
- Aberto: GTBS, Drib, SD, CP e PE usam o mesmo facing cru + bônus 100; a mesma diluição é provável.

## GoToBallShoot com o mesmo fix (tier 0, 500k passos, seeds 0-1)

Sonda determinística, 20 episódios (`probe_gtbs.py`): menor distância à bola, fração que chega
(< 0.13 m), chutou, bola fora, gol.

| variante | sucesso treino | dist. mín. (mediana) | chega | chutou | fora | gol |
|---|---|---|---|---|---|---|
| HEAD (facing cru, bônus/fora 100) | 0 / 0 | 0.64 / 1.45 m | 0.05 / 0 | 0 / 0 | 0.05 / 0 | 0 |
| facing potencial, bônus 10, fora −10 | 0.02 / 0.03 | 0.18 / 0.14 m | 0.40 / 0.50 | 0 / 0.20 | 0 / 0.10 | 0 |

HEAD acampa (ep_rew ~10-12 ≈ facing cru acumulado) e nunca vai até a bola. O fix leva até a bola,
mas a política passa só 0.2-2.4 passos por episódio em `_at_ball` (< REACH e cos > 0.9), e o chute
só conta ali. Fix levado a `_KickTask` (GTBS, SD, PE; o cos β do PE entra no mesmo potencial);
`test_kick_tasks_do_not_pay_camping`: girar no lugar rendia 70.1 / 117.8 / 44.1, agora < 1.
Próximo gargalo: assentar na bola de frente.
