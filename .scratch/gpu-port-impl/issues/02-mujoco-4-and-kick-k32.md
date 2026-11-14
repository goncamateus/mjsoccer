# 02: MuJoCo 4.0 no backend CPU, com kick K = 32

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A base sobe de `mujoco 3.11` para `mujoco 4.0.0` (avaliar depender só de `mujoco-core`). O passo de disparo vira K = 32: 32 passos de física a `dt/4` + 16 a `dt`, 48 iterações, o passo de controle segue com 25 ms (`FIRING_SUBSTEPS = 4`, novo `FIRING_FINE_STEPS = 32`). A `KICK_TABLE_MS`, chave só por plunger, é recalibrada no CPU 4.0 float64 com K = 32 a 40 Hz pelo `calibrate.py` do candidate-09 adaptado. Os testes de fidelidade do `physics-fidelity` rodam de novo no 4.0 (o 4.0 muda derivadas de atuador saturado em `implicitfast` e `nativeccd`/`multiccd`).

**Blocked by:** 01 (Remover o tier 1)

**Status:** done

- [x] `pyproject.toml` pina `mujoco 4.0.0`; `uv sync` limpo
- [x] Passo de disparo com 48 iterações (32 a `dt/4`, 16 a `dt`) e duração de 25 ms
- [x] `KICK_TABLE_MS` indexada só por plunger, recalibrada no 4.0; velocidade de saída do kick e do chip dentro dos critérios
- [x] Testes de fidelidade (envelope, bola, posse, kick/chip) verdes no 4.0, `-m validation` incluso
- [x] Base CPU de decisões/s remedida no 4.0 (ContestedPossession 24 envs, GoToBall 12 envs) e anotada neste ticket

## Comments

**Feito em 2026-11-15.**

- **Dependência.** `pyproject.toml` pina `mujoco==4.0.0`; `uv sync` limpo. `mujoco-core` sozinho
  não serve: a base usa `mujoco.viewer` e `mujoco.Renderer` (que no core exigem o extra
  `rendering`), e o ticket 06 (viewer Studio) precisa de `mujoco-studio`, que só o `mujoco` cheio
  traz. Fica o `mujoco`.
- **Disparo K = 32.** `scene.FIRING_FINE_STEPS = 32` a `dt / FIRING_SUBSTEPS`, depois
  `decimation − 32/4` passos a `dt`: a 40 Hz, 32 + 16 = 48 iterações em 25 ms. Teste novo
  `test_firing_step_is_48_iterations_over_one_control_step`. A 60 Hz (default do env até o ticket
  04) são 32 + 8.
- **Tabela.** `calibrate.py` do candidate-09 adaptado (40 Hz, sem tier, 3 casas); saída em
  `.scratch/physics-fidelity/candidate-09/calibrate_mj4_k32.txt`. Bate com o K = 32 do
  `15-kick-on-warp.md` (125 N: 4.244 × 4.246). O teto de 6.5 m/s cai em ~314 N (kick) e ~295 N
  (chip). `test_kick_exit_speed` agora cobre a ∈ {0.25, 0.5, 0.75, 1.0} × kick/chip, ±0.1 m/s a
  40 Hz (com a tabela antiga falhava: 1.74 × 1.625, 3.44 × 3.25, chip a 1.0 em 6.25) e ±0.3 a 60 Hz.
- **Fidelidade.** `uv run pytest -m "validation or not validation"`: 99 passed no 4.0 (envelope,
  COR/inércia da bola, posse, kick/chip, `spin`, `kick_recoil`, random, competent).
- **Decisões/s** (CPU, 40 Hz, ação aleatória, autoreset, melhor de 3 × 20 s, máquina ociosa;
  `.scratch/gpu-port-impl/02-bench/dps.py`, saídas `dps_mj4_k32.txt` e `dps_mj311_k96.txt`).
  Remedido no mesmo dia o 3.11 + K = 96 do `gpu-port-impl` (antes deste ticket) para comparar:

  | | 3.11, K = 96 | 4.0, K = 32 | × |
  |---|---|---|---|
  | ContestedPossession, 24 envs `AsyncVectorEnv` | 1 513 | **4 862** | 3.2 |
  | ContestedPossession, 1 env sync | 526 | **1 080** | 2.1 |
  | GoToBall, 12 envs `AsyncVectorEnv` | 319 | **1 802** | 5.6 |
  | GoToBall, 1 env sync | 203 | **467** | 2.3 |

  O 3.11 reproduz a base do issue 06 (1 535 / 299 / 531 / 205). A nova base CPU para o MJX é
  **4 862 DPS (2 robôs) e 1 802 DPS (7 robôs)**; a folga do Warp cai para ~8× / ~2.6× sobre a
  física pura do 05. Não separei o ganho do 4.0 do ganho do K = 32 (ação aleatória dispara a cada
  re-arme, então o K pesa aqui mais que numa política).
