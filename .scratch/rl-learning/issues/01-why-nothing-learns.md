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
| GoToBall | **reward novo, frame-skip 4** | **0.95** | **0.98**, episódio ~294 passos (scripted 285-307) |

## Fixes

- `e0e59c5` — GoToBall: facing vira potencial, `0.5 * (facing − prev_facing)`; girar no lugar
  rende 0.27. `test_goto_ball_does_not_pay_camping`.
- `9f7d913` — ticket `physics-fidelity/22`: contato bola↔rolete de PU 65A; limite de giro com
  bola ~1 → ~2 rad/s.

## Pendente

- Seeds 1-4 de GoToBall reward novo + frame-skip 4, e reward antigo + frame-skip 4 (seed 0),
  para separar o efeito do fix do reward do efeito do frame-skip. Rodando.
- Levar o frame-skip ao `examples/sac_agent.py` (`--control-hz`).
- Shoot: frame-skip 4 + peso da aproximação 0.2 → 1.0.
- Dribbling: retrainar depois do ticket 22.
