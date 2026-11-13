# 04: Ação em target pose e controlador de navegação no backend CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A ação vira `(x, y, sin θ, cos θ, kick, chip, dribble)` nas 6 tasks (`x, y` × 6.0 × 4.5 m, velocidade-alvo 0). O `go_to_point_new` do Planning-the-path-with-rl (`2287531`) roda como função pura, fixo a 40 Hz (sai `control_hz`), com os quirks verbatim (`MAX_DV = 0.1` por passo, frenagem a `MAX_ACCEL = 1`, sem banda morta de posição, zona morta de 2.5°, `ω = 2·Δθ` sem limite) e `max_velocity = 3.3`. Entrada pelo `qpos`/`qvel` das juntas do robô; saída pela rotação body→world até os servos. A política decide a cada `repeat_action` passos de controle (default 1). Só os `n_controlled` passam pelo controlador; os scripted ficam como hoje; no PassEndurance `vx, vy` vão a 0 depois do controlador. Um caminho interno de velocidade crua, fora do `action_space`, fica para os testes de envelope. `goto_ball_policy` devolve target pose (alvo = bola, heading robô→bola); testes de env, `make_gifs` e `validation_sweep` adaptados. Observação inalterada.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** ready-for-agent

- [ ] `action_space` com 7 dims em todas as tasks
- [ ] Controlador puro, sem mutação de estado, com os quirks do original e `max_velocity = 3.3`
- [ ] Passo de controle de 25 ms (24 passos de física); kwarg `repeat_action` funciona; `control_hz` não existe mais
- [ ] PassEndurance: o robô só gira, `x, y` da ação ignorados
- [ ] Testes de envelope rodam pelo caminho de velocidade crua e continuam verdes
- [ ] `goto_ball_policy` em target pose; `uv run pytest` verde; `make_gifs` e `validation_sweep` rodam
