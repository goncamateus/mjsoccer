# 04: Ação em target pose e controlador de navegação no backend CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A ação vira `(x, y, sin θ, cos θ, kick, chip, dribble)` nas 6 tasks (`x, y` × 6.0 × 4.5 m, velocidade-alvo 0). O `go_to_point_new` do Planning-the-path-with-rl (`2287531`) roda como função pura, fixo a 40 Hz (sai `control_hz`), com os quirks verbatim (`MAX_DV = 0.1` por passo, frenagem a `MAX_ACCEL = 1`, sem banda morta de posição, zona morta de 2.5°, `ω = 2·Δθ` sem limite) e `max_velocity = 3.3`. Entrada pelo `qpos`/`qvel` das juntas do robô; saída pela rotação body→world até os servos. A política decide a cada `repeat_action` passos de controle (default 1). Só os `n_controlled` passam pelo controlador; os scripted ficam como hoje; no PassEndurance `vx, vy` vão a 0 depois do controlador. Um caminho interno de velocidade crua, fora do `action_space`, fica para os testes de envelope. `goto_ball_policy` devolve target pose (alvo = bola, heading robô→bola); testes de env, `make_gifs` e `validation_sweep` adaptados. Observação inalterada.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** done

- [x] `action_space` com 7 dims em todas as tasks
- [x] Controlador puro, sem mutação de estado, com os quirks do original e `max_velocity = 3.3`
- [x] Passo de controle de 25 ms (24 passos de física); kwarg `repeat_action` funciona; `control_hz` não existe mais
- [x] PassEndurance: o robô só gira, `x, y` da ação ignorados
- [x] Testes de envelope rodam pelo caminho de velocidade crua e continuam verdes
- [x] `goto_ball_policy` em target pose; `uv run pytest` verde; `make_gifs` e `validation_sweep` rodam

## Comments

**Feito em 2026-11-14** (branch `gpu-port-04-nav-controller`).

- **Controlador.** `src/mjsoccer/navigation.py::go_to_point`, função pura. Testado bit a bit
  (atol 1e-12, 5 000 estados aleatórios, incluindo yaw desenrolado e chegada) contra uma cópia
  do `go_to_point_new` original em `tests/_ptp_navigation.py` (2287531, verbatim, só oráculo de
  teste), mais testes dos quirks: `MAX_DV`, frenagem a 1 m/s², teto 3.3, saída no corpo, zona
  morta de 2.5°, `ω = 2Δθ`, pureza. O clone de leitura do `map.md` em `/tmp` tinha sumido; peguei
  `envs/navigation.py` e `envs/enhanced.py` do GitHub no commit `2287531e8`.
- **Seams testados** (sem usuário para confirmar, escolhi): `navigation.go_to_point`; a API do env
  (`step`, `action_space`, kwarg `repeat_action`); o caminho interno `BaseEnv.step_raw`.
- **Env.** `CONTROL_DT = 0.025`, `decimation = 24` fixo; `control_hz` sai (vira `TypeError`).
  `step` = uma decisão = `repeat_action` passos de controle, **recompensas somadas**, para no
  término/truncation (o original devolvia só a última; somar mantém o shaping por passo).
  `steps`/`max_steps`/lock/grace em passos de controle; `render_fps` = decisões/s.
  Entradas do controlador = `qpos + xy0` e `qvel` das juntas; saída pela rotação `xmat` de
  sempre. `_shape_drive` é o hook de task; o PassEndurance devolve `(0, 0, ω)`.
- **Energia** (`_ShootTask._energy`): era `|ação[:3]|`; agora é o comando do controlador como
  fração do `ctrlrange` (`BaseEnv.drive_cmd`), recortado em ±1.
- **`step_raw`.** Fatia `(vx, vy, ω, kick, chip, dribble)`, body frame × `ctrlrange`, direto
  nos servos. Usado por `spin`/`kick_recoil` do `validation_sweep` (flag nova na tupla de
  `SCENARIOS`), pelo teste do giro a 2 rad/s e pelo `test_kick_exit_speed`.
- **Achado — kick e chatter.** Segurando a pose pelo controlador, o chatter de ±0.1 m/s (sem
  banda morta) mexe o chassi a ~5 cm/s e a bola sai 0.1-0.3 m/s abaixo da tabela (a = 1: 6.37
  contra 6.55 com os servos em zero). A tabela é calibrada com o robô parado, então o teste de
  calibração usa `step_raw`. Política treinada que chuta parada no alvo vai ver isso.
- **Ação zero** agora é uma target pose (centro, olhando +x). Testes longos que queriam “parado”
  usam `_hold(env)` (pose atual).
- **Medidas a 40 Hz:** `goto_ball_policy` 20/20 no GoToBall (seeds 0-19); posse assentada no
  Dribbling 119/120 segurando a pose; giro de 2 rad/s 78/120 em todas as seeds (barra 77 → 74,
  agora 120 passos são 3 s).
- **Verificação:** `uv run pytest` 87 passed; `-m "validation or not validation"` 106 passed;
  `validation_sweep --seeds 0-4` e `make_gifs` (para o scratchpad, GIFs do repo intactos) rodam.
  `examples/*` trocam `--control-hz` por `--repeat-action`; modelos SB3 salvos com a ação de 6
  dims não carregam mais.
- **Depois do merge com `gpu-port-impl`** (tickets 03 e 06 dentro): o
  `test_possession_reads_no_contact_list` do 03 usava ação zero; passou a usar `_hold`.
  `uv run pytest` 94 passed, `-m validation` 19 passed.
