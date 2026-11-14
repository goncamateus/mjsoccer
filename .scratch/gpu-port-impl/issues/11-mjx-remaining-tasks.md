# 11: As quatro tasks restantes no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** GoToBallShoot, Dribbling, StaticDefenders e PassEndurance rodam no MJX-Warp, com os robôs scripted como no CPU (amarelos com `ctrl = 0`, companheiro do PassEndurance só com dribbler) e `vx, vy` zerados no PassEndurance. As 6 tasks ficam cobertas.

**Blocked by:** 10 (GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s)

**Status:** done

- [x] `mjsoccer.mjx.make` funciona para todos os ids do `ENV_IDS`
- [x] Task-logic parity e Warp smoke cobrem as 6 tasks
- [x] `uv run pytest` verde

## Comments

**Feito em 2026-11-14** (branch `gpu-port-11-mjx-remaining`).

- **As 4 tasks** em `mjx/tasks.py`, uma classe por classe do CPU, config lida do `cpu_cls`:
  - `GoToBallShootEnv` (`_ShootTask`): bola e robô uniformes na metade de ataque; o robô é
    resorteado enquanto estiver a < `sep` da bola nos dois eixos (Chebyshev). 8 candidatos; o
    primeiro y sai negado, como no CPU (`-u(*ys)`).
  - `StaticDefendersEnv` (`_ShootTask`): robô na origem, bola resorteada fora da área do goleiro
    (8 candidatos; ~13 % de rejeição, todos falham ~1e-7), depois os 6 amarelos pelo `scatter`.
  - `DribblingEnv`: posições fixas, bola assentada no azul (`seat_ball`), estado `count` e
    `last_ball`, `task_obs` = `(count/6)·2 − 1`. A cadeia `elif` do CPU virou máscaras.
  - `PassEnduranceEnv`: `shape_drive` zera `vx, vy`, `script` liga o dribbler do recebedor,
    estado `prev`/`stopped`, `_wrong_ball` em centímetros truncados (`astype(int32)`, como o
    `int` do numpy). O x do recebedor é resorteado enquanto `|rx − bx| < 1` numa faixa de 3 m:
    até 2/3 de rejeição, então `PASS_CANDIDATES = 64` ((2/3)^64 ≈ 5e-12).
  - Amarelos sem comando (`ctrl = 0`) em todas, como no CPU.
- **`first_where(cand, free, roomy)`** (`mjx/tasks.py`): o `first_free` para qualquer teste de
  rejeição; nenhum livre → o de maior `roomy`, `overlap` → `reset_overlap`.
- **Parity** (`test_mjx_task_logic.py`): as 6 em `TASKS` (place nos mesmos candidatos com o
  caminho de rejeição forçado, init, rollouts seeds 0-4). Novo
  `test_every_env_id_is_ported_and_covered`: `TASKS` == `ENV_IDS`, e o `make` de cada id bate
  obs/action/`max_steps` com o CPU.
  - O `_Scattered` ganhou blocos sem heading (`lead`) encerrados por `end_block()`; no
    StaticDefenders um hook em `tasks._in_gk_area` encerra o bloco da bola quando o CPU aceita.
  - Rollouts: o limite de 300 passos virou `cpu.max_steps` (StaticDefenders não terminava em
    300 com a perseguição).
  - **Ramos que rollout não alcança**, teleportando a bola e comparando com o `_reward_done` do
    CPU (`_ball_walk`): Dribbling — zigue-zague até 7 checkpoints, cruzamento de volta no último
    gap, 600 pontos aleatórios; PassEndurance — bola parada (> 20 passos), recebedor com a bola
    (+1), bola fora da caixa. Os rollouts do PassEndurance sempre acabam por bola fora da caixa
    e os do Dribbling nunca passam um checkpoint. Mutações (`c == 7` → `5`, `stopped >` → `>=`,
    recompensa recebida 1 → 0.5) derrubam os testes.
- **Warp smoke** (`test_mjx_warp_smoke.py`): as 6 em `TASKS`, repeat_action 1 e 2; posse do azul
  no reset no Dribbling e PassEndurance; espaçamento ≥ 0.2 m também no StaticDefenders. Novo
  `test_warp_pass_endurance_scripted_robots`: no Warp, o `ctrl` das slides do passador é 0
  qualquer que seja a ação e o dribbler do recebedor gira (mutação: sem `shape_drive`, falha).
  NaN/overflow/kick seguem só no ContestedPossession. Runs pequenos: 4 mundos, 10 passos,
  ~8-12 s cada.
- Não medi decisões/s nem `naconmax` das 4 novas (GPU reservada ao 13); o sizing do 10
  (`max(64, 24 × n_robots)`) vale, e nenhum smoke acusou bit de overflow.
- **Verificação** (`git merge gpu-port-impl`: já em dia): `uv sync && uv run pytest` 95 passed,
  2 skipped; `uv sync --group gpu` + `systemd-run --user --scope -p MemoryMax=10G uv run pytest`
  214 passed. Nenhum processo na GPU depois.
