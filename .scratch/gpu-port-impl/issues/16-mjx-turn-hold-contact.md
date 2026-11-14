# 16: Posse no giro de 2 rad/s no MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** No giro de 2 rad/s com a bola (`hold_turn`), o MJX-Warp mantém a bola 69–78 de 120 passos (mediana 73), não determinístico do mesmo estado; a CPU mantém 78 sempre, limiar 74 (ticket 12, `## Comments`). Investigar o contato bola × dribbler em float32 no Warp (causa do não determinismo e da perda de posse) e corrigir só com física/solver legítimos, sem cheats. Decisão do usuário (2026-11-14): investigar antes de aceitar limiar próprio do MJX.

**Blocked by:** 12 (Paridade MJX-Warp × CPU)

**Status:** done (opção 2 do usuário: 74 na trajetória exata da CPU; mediana de 64 cópias perturbadas ≥ 69 nos dois backends)

- [x] Causa do não determinismo e da perda de posse identificada e registrada aqui
- [x] MJX passa o limiar da CPU (74) no `hold_turn`, ou o usuário aceita um limiar MJX com os números medidos
- [x] `xfail` do caso MJX removido; `uv run pytest -m validation` com o grupo `gpu` verde (153 passed, 11.4 min)

## Comments

**Investigado em 2026-11-14** (branch `gpu-port-16-turn-hold`). Sondas em
[`16-probes/`](16-probes/) (`uv run [--group gpu] python .scratch/gpu-port-impl/issues/16-probes/<x>.py`).

- **Causa: o giro com a bola é caótico, também na CPU.** Os 78/120 da CPU são uma trajetória
  só, reproduzida bit a bit porque o MuJoCo CPU é determinístico; uma perturbação física
  desprezível do mesmo estado (seed 0, `qvel += eps * N(0,1)`, `cpu_loop.py`) já espalha a
  posse da CPU:

  | eps em `qvel` (m/s, rad/s) | posse /120 na CPU (40 amostras) |
  |---|---|
  | 0, 1e-12, 1e-9 | 78 em todas |
  | 1e-7 | 67-78, mediana 77.5 |
  | 1e-5 | 64-80, mediana 73 |
  | 1e-3 | 67-83, mediana 73.5 |

  O limiar de sensibilidade (1e-7..1e-5) é a escala do float32 (eps relativo 1.2e-7; a barra
  gira a ~314 rad/s com `dribble = 1`, erro absoluto ~4e-5). Prova direta: a CPU em float64 com só o
  `qpos`/`qvel` arredondado para float32 depois de cada `mj_step` (`cpu_f32.py`) segura
  **69**, não 78. O stick-slip da barra (agarra, escorrega, reagarra) amplifica qualquer
  diferença de arredondamento até decidir em que passo a bola escapa.
- **O não determinismo do Warp é só a semente desse caos.** O Warp na GPU não é
  determinístico (ordem dos contatos por atomics, FAQ do MJWarp; pesquisa 01), então cada
  cópia do lote recebe um arredondamento diferente e cai num ponto diferente da mesma
  distribuição. Um Warp determinístico em float32 daria uma amostra fixa dela, não 78.
- **O MJX não perde mais a bola que a CPU — segura um pouco mais.** 64 cópias do mesmo
  estado no MJX (`hold_loop.py 64`) contra 64 perturbações eps = 1e-5 na CPU:

  | | n | média | mediana | ≥ 74 | min-max |
  |---|---|---|---|---|---|
  | CPU perturbada | 64 | 71.95 | 72 | 28 % | 66-78 |
  | MJX-Warp | 64 | 73.69 | 73.5 | 50 % | 68-80 |

  Mann-Whitney p = 0.003 (MJX ~1.7 passo acima). Não há perda de posse a corrigir no
  contato bola × barra do Warp: a distribuição é a da física, e a da CPU fica abaixo.
- **Solver só no MJX não muda nada** (32 cópias cada, `hold_loop.py 32 <opt>=<v>`): padrão
  mediana 73-73.5; `iterations=300 ls_iterations=100` 68-81, mediana 74; `tolerance=1e-12`
  69-80, mediana 74; `tolerance=1e-5` 69-80, mediana 75 — o mesmo espalhamento, dentro do
  ruído amostral. Noslip não existe no Warp; float64 não existe no Warp. Mexer no contato
  (solref/atrito/geometria) mudaria a CPU e a fidelidade medida (tickets 17/22): fora do
  escopo e proibido aqui. **Nenhum ajuste legítimo leva o MJX a ≥ 74 de forma confiável,
  porque nem a CPU passa 74 de forma confiável fora da sua única trajetória.**
- **O que está mal posto é o critério**: 74 = 5 % abaixo de uma amostra da CPU, numa métrica
  cujo espalhamento físico é ±6 passos. Num mundo só, o MJX passa ~50 % das vezes; a CPU
  passa sempre só por ser determinística (perturbada, 28 %). Não mudei o limiar nem o
  critério. Opções para o usuário decidir:
  1. Critério por distribuição nos dois backends: N cópias perturbadas (CPU: eps 1e-5 em
     `qvel`; MJX: o próprio lote) e mediana ≥ um limiar tirado da CPU perturbada (p. ex. 70,
     ~2 passos abaixo da mediana 72); a mediana fica estável (±1 passo) com N = 32-64.
  2. Manter 74 só na CPU (trajetória fixa, regressão bit a bit) e no MJX usar o critério 1.
  3. Aceitar o `xfail` como está (relatório).
- `xfail` do caso MJX mantido, motivo atualizado no `test_validation.py`.

**Status → needs-decision.**

### Decisão do usuário (2026-11-14): opção 2 — implementada

- **CPU**: `test_seated_ball_survives_a_2_rad_s_turn` só na CPU, 74/120 na trajetória exata
  (regressão bit a bit; 78 medido).
- **Os dois backends**: cenário novo `hold_turn_noisy` no `validation_sweep.py` — o mesmo
  giro, `HOLD_TURN_COPIES` = 64 cópias, a cópia k com `HOLD_TURN_NOISE` = 1e-5 · N(0, 1)
  (seed k) somado a todo o `qvel` antes do primeiro passo; no MJX cada cópia é um mundo do
  lote, com o estado da CPU. `test_seated_ball_turn_over_perturbed_copies[cpu|mjx]`: mediana
  da posse ≥ `HOLD_TURN_MEDIAN_MIN` = **69/120**. Rodar também na CPU mantém o mesmo limiar
  nos dois backends e mostra que o critério é satisfazível pela física de referência.
- **Escolha do limiar** (CPU perturbada, medianas de 64 cópias em cinco sorteios de ruído:
  72, 71, 72, 71, 73; agrupada ~72): 69 = 72 − 5 %, a mesma folga do 74 sobre o 78 exato.
  Fica 2 abaixo da pior mediana de 64 da CPU (71) e 4.5-5 abaixo do MJX.
- **Medido no cenário final**: CPU 64 cópias mediana 72 (64-80); MJX-Warp mediana 74
  (64-83), e 73.5 numa medição anterior de 64 cópias. A CPU é determinística, então o caso
  `cpu` dá sempre 72; o MJX oscila ~±1 em mediana de run para run, com ≥ 4 de folga.
- `xfail` removido. `docs/index.md` atualizado.
