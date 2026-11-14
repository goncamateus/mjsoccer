# 03: Posse pelo sensor `contact` no backend CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A posse deixa de varrer `data.contact` e lê um sensor `contact` geom-geom (bola × barra do dribbler), a mesma definição que o MJX vai ler. O sensor não altera a descrição física do modelo base.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** done

- [x] Posse calculada só a partir do sensor `contact`; nenhuma varredura de `data.contact` na lógica da task
- [x] Geometria, massas, contato e atuadores do modelo base inalterados
- [x] Testes de posse (bola assentada, giro de 2 rad/s) verdes

## Comments

- `scene.build` adds, per robot, two `contact` sensors geom-geom `ball` x `{prefix}r_col_barDribbler` and `ball` x `{prefix}r_col_axisDribbler`, `data="found"`, `num=1` (forma da pesquisa 01 §6). O eixo entra junto da barra porque é o outro geom colidível do corpo `dribbler`; só a barra mudaria a definição antiga. `RobotIds.touch_adr` guarda os índices em `sensordata`, `RobotIds.touches_ball(data)` lê.
- `BaseEnv._update_possession` só lê `touches_ball`; nenhum `data.contact` em `src/`.
- Testes novos (`tests/test_envs.py`): o sensor bate passo a passo com a varredura antiga de `data.contact` (2 robôs, 3 seeds x 2 portadores x 600 passos, bar e giro aleatórios); modelo com e sem sensores (`build(..., possession_sensors=False)`) tem geoms, massas, pares, atuadores e damping idênticos e rollout de 500 passos bit a bit igual; `holds()` funciona com `data.contact`/`ncon` bloqueados.
- `test_seated_ball_is_possessed` e `test_seated_ball_survives_a_2_rad_s_turn` verdes sem mudar limiares. `uv run pytest`: 83 passed; `-m validation`: 19 passed.
