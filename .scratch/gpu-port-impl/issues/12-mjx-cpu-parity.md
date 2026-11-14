# 12: Paridade MJX-Warp × CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `validation_sweep.py --backend {cpu,mjx}`: cenários definidos uma vez; no MJX cada (cenário, param, seed) é um mundo do lote; CSVs com coluna `backend`; import do MJX preguiçoso. Em `test_validation` os testes de critério ganham o parâmetro `backend` (MJX passa os mesmos limiares da CPU) e entram `test_parity_deterministic` (≤ 2 % em envelope, kick e COR; ≤ 5 % no tempo de chegada; ≤ 1 cm / 1° no erro de chegada, mesmo estado inicial em `qpos`/`qvel`) e `test_parity_seeded` (mediana MJX no [p25, p75] da CPU, seeds 0-4, amostras simétricas). Casos MJX pulados sem CUDA. Comandos documentados no `docs/index.md`, com o sweep sob `systemd-run --user --scope -p MemoryMax=...`.

**Blocked by:** 05 (Critérios de aceitação do controlador de navegação), 11 (As quatro tasks restantes no MJX)

**Status:** done (a caixa do giro de 2 rad/s no MJX segue no ticket 16)

- [x] `validation_sweep.py --backend mjx --seeds 0-4` roda e grava CSVs com coluna `backend`
- [ ] Todos os critérios passam no MJX com os limiares da CPU — **todos menos o giro de 2 rad/s com a bola: movido para o ticket 16**
- [x] `test_parity_deterministic` e `test_parity_seeded` verdes nesta máquina — **seeded com 50 seeds no MJX (ver Comments)**
- [x] Comandos de teste por camada documentados nos docstrings e no `docs/index.md`

## Comments

**Feito em 2026-11-14** (branch `gpu-port-12-mjx-parity`).

- **Sweep** (`scripts/validation_sweep.py`): `--backend {cpu,mjx}`; `units()` define os
  episódios uma vez; `run(units, backend)` = pool de processos no CPU, um lote MJX-Warp por
  (env, caminho raw) no MJX, cada (cenário, param, seed) um mundo. Cenários scripted: o
  set-up roda no env CPU e `qpos`/`qvel`/`xfrc_applied` vão para o mundo (mesmo estado
  inicial); cenários com seed: `reset(PRNGKey(seed))` no MJX, política sobre a obs do mundo.
  Linha do CSV (`_row`) calculada igual nos dois a partir do traço; coluna `backend` em
  todos os CSVs. Import do MJX só no ramo `mjx`; sem grupo `gpu`/CUDA o script sai 77.
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` por default. Linhas MJX não têm `impulse`,
  `success` nem motivo além de terminated/truncated (o step jitado não devolve `info`).
- **Raw no MJX**: `MjxEnv.control(..., raw=True)` / `step_raw` (o envelope passa por fora do
  controlador, como o `step_raw` do CPU); coberto no task-logic parity
  (`test_raw_control_matches_cpu`, ContestedPossession e PassEndurance).
- **Cenários novos** (determinísticos, uma vez por parâmetro), porque o `test_validation` não
  tinha envelope/kick/COR/posse pela seam do sweep: `envelope` (vx/vy/ω cheio do repouso,
  3 s), `kick` (kick/chip a = 0.5/1, bola assentada), `bounce` (chão/parede a 1/6.5 m/s,
  gravidade cancelada por `xfrc_applied`, COR lido 0.2 s depois), `hold_turn` (2 rad/s com
  a bola). Inércia: `ball_beta` lido do modelo de cada backend (`env.sys` no MJX).
  Limiares iguais aos testes do CPU (`test_envs.py`): envelope ±3 %, kick ±0.1 m/s, COR
  0.47-0.56, posse ≥ 74/120.
- **Achado — `mjx.forward` duas vezes no mesmo jit falha no Warp**: reset + reescrita do
  estado + `mjx.forward` num só `jit(vmap(...))` dá `Failed to allocate 264 bytes` dentro da
  captura do grafo (alocação no `convex_narrowphase`). Separado em dois jits, roda.

**Paridade medida** (RTX 3060, sweep completo, seeds 0-4):

| cenário | métrica | CPU | MJX | diferença |
|---|---|---|---|---|
| envelope vx / vy | aceleração m/s² | 1.3074 / 1.3076 | 1.3075 / 1.3075 | 0.008 % |
| envelope vx / vy | v final m/s | 3.3000 / 3.2916 | 3.3000 / 3.2916 | 0 |
| envelope ω | α / ω final | 55.063 / 37.066 | 55.063 / 37.066 | 0.0003 % |
| spin k=0.2..1 | yaw p99 | 7.419 .. 37.078 | idem | 0 |
| kick a=.5/1 | m/s | 3.2259 / 6.5525 | 3.2259 / 6.5525 | 0 |
| chip a=.5/1 | m/s | 3.2823 / 6.4592 | 3.2823 / 6.4592 | 0 |
| COR chão 1 / 6.5 | | 0.5463 / 0.5546 | 0.5463 / 0.5546 | 0 |
| COR parede 1 / 6.5 | | 0.5472 / 0.5319 | 0.5472 / 0.5319 | ≤ 0.009 % |
| navigate (9 células) | tempo de assentamento | 1.125 / 2.45 / 4.375 s | 1.125 / 2.425-2.45 / 4.375 | ≤ 1.02 % (tol 5 %) |
| navigate | erro final de posição | 0.25-1.85 mm | 0.10-2.14 mm | ≤ 1.47 mm (tol 1 cm) |
| navigate | erro final de heading | 0.008-2.37° | 0.016-2.43° | ≤ 0.14° (tol 1°) |
| navigate | chatter RMS | 0.0347-0.0444 | 0.0339-0.0424 | ≤ 17 % (sem tolerância de paridade) |
| hold_turn | posse /120 | 78 | **69-78, mediana 73** | ver abaixo |

  Critérios do controlador no MJX: pior heading 2.43° (< 2.5), pior chatter 0.0424 (< 0.05),
  tempo/trapézio ≤ 0.95, ω comandado ≤ 2π. O chatter do MJX difere até 17 % célula a célula
  (o empurrão lateral em repouso do float32, como o 07 previu), mas fica no limiar.

- **`hold_turn` no MJX — critério NÃO passa de forma confiável**. O Warp não é determinístico
  de run para run: 16 cópias do mesmo estado inicial num lote seguraram a bola 69-78 de 120
  passos (mediana 73), duas vezes; em runs separados do sweep, 74, 75, 71. O CPU segura 78
  sempre, e o limiar é 74 (5 % abaixo). Não afrouxei: o caso `mjx` do
  `test_seated_ball_survives_a_2_rad_s_turn` está `xfail(strict=False)` com os números. A
  decidir: aceitar a posse do MJX no giro (média ~6 % abaixo) com outro limiar, ou investigar
  o contato bola/barra em float32. Os outros critérios não mostraram variação entre runs
  (determinísticos batem com o CPU a ≤ 0.01 %).
- **`test_parity_seeded` revisado depois da primeira medição** (o spec permite): o `[p25, p75]`
  planejado reprova distribuições idênticas metade das vezes — com 5 amostras de cada lado,
  a mediana de uma cai no IQR da outra com probabilidade 0.48 (simulado, percentis
  interpolados do numpy). Medido, reprovou 5 de 24 células, todas do `competent`
  (ContestedPossession posse 0.0169 vs [0.0109, 0.0168]; GoToBallShoot retorno −0.041 vs
  [−0.731, −0.052] e posse 0.336 vs [0.370, 0.636]; PassEndurance retorno −0.9999 vs
  [−0.9996, −0.9996]; StaticDefenders retorno 0.523 vs [0.138, 0.315]). O teste agora usa o
  [min, max] da CPU (0.83 de aprovação por célula sob a hipótese nula: é checagem de
  divergência grosseira, não teste de distribuição). **Ainda instável**: no primeiro run as
  24 passaram; no `-m validation` final, `competent` StaticDefenders retorno deu mediana MJX
  0.5273 contra máx CPU 0.5252 (0.5226 no run anterior) e reprovou. O Warp não é
  determinístico de run para run, e com 5 seeds qualquer intervalo da CPU reprova uma
  fração fixa das células mesmo com distribuições iguais. Não afrouxei mais; a decidir:
  mais seeds no MJX (barato: são mundos do lote; o CPU fica em 0-4), comparar só o
  agregado entre envs, ou aceitar o teste como relatório.
- **Custo**: sweep MJX completo ~10 min (compilação por lote + laço no host passo a passo;
  o Dribbling a 4 800 passos domina); `-m validation` com o grupo `gpu` ~12.5 min (CPU + MJX
  em subprocessos). Pico de GPU ~0.5 GB.
- **Verificação**: `git merge gpu-port-impl` em dia. Com o grupo `gpu`: `uv run pytest`
  216 passed; `-m validation` 149 passed, 1 xfailed (`hold_turn[mjx]`, 73/120 nesse run),
  1 failed (`test_parity_seeded[competent-StaticDefenders-v0-ret]`), 12.7 min. Sem o grupo:
  `uv run pytest` 95 passed, 2 skipped; `-m validation` 41 passed, 112 skipped (MJX), 47 s. Nenhum processo na GPU depois.

### Decisão do usuário (2026-11-14)
- Giro de 2 rad/s no MJX: investigar o contato → ticket 16.
- `test_parity_seeded`: mais seeds só no MJX (mundos extras são baratos), CPU fica em seeds 0-4; fixar o número de seeds pela taxa de falso-negativo simulada.

**`test_parity_seeded` com mais seeds no MJX — 2026-11-14** (branch `gpu-port-16-turn-hold`).

- **Só mais seeds no MJX não basta com a regra antiga.** Simulado (200 000 tentativas, as duas
  amostras da mesma distribuição contínua): "mediana MJX no [min, max] da CPU de 5 seeds"
  falha por célula 0.166 / 0.117 / 0.091 / 0.075 / 0.069 com 5 / 10 / 20 / 50 / 100 seeds
  MJX, e nunca abaixo de 2/32 = 0.0625 (a mediana verdadeira cai fora do intervalo de 5
  sorteios 2 × 0.5⁵ das vezes) — 82 % de chance de alguma das 24 células falhar mesmo com
  100 seeds. O gargalo é o intervalo de 5 seeds da CPU, não a mediana do MJX.
- **Regra trocada de lado**: a **mediana da CPU** (seeds 0-4) dentro do **[min, max] do MJX**
  (`MJX_SEEDS`). Agora a taxa cai com as seeds do MJX. Simulado, falha por célula / chance de
  alguma das 24 falhar com distribuições iguais:

  | seeds MJX | célula | família (24) | célula com desvio de 3 σ |
  |---|---|---|---|
  | 10 | 0.044 | 0.66 | 0.96 |
  | 20 | 0.0087 | 0.19 | 0.93 |
  | 30 | 0.0031 | 0.072 | 0.90 |
  | 40 | 0.0015 | 0.035 | 0.88 |
  | **50** | **0.00074** | **0.0175** | **0.86** |

  Alvo: família ≤ 2 % → **50 seeds no MJX (0-49)**, a menor contagem da grade que fica abaixo.
  Continua sendo checagem de divergência grosseira (um desvio de 3 σ reprova a célula 86 %
  das vezes, 1 σ só 3 %), não teste de distribuição. Empates só ajudam (intervalo inclusivo).
- **Código** (`tests/test_validation.py`): `MJX_SEEDS = range(50)`; o fixture `sweep` roda o
  MJX com `--seeds 0-49` e a CPU com `0-4`; os critérios por seed (`random`, `competent`)
  continuam em `SEEDS` (0-4) nos dois. `_seeded_parity_ok` é a regra;
  `test_mjx_seed_count_holds_the_false_failure_rate` refaz a simulação (rápido, < 1 s) e
  prende `len(MJX_SEEDS)` à menor contagem da grade (10..60) com família ≤
  `SEEDED_FAMILY_FALSE_FAIL` = 2 %, e exige > 0.8 de detecção a 3 σ. `docs/index.md`
  atualizado.
- **Medido**: `-m validation -k "parity_seeded or seed_count"` com o grupo `gpu`: 25 passed
  (24 células + a simulação). Custo: o sweep MJX passou de ~10 para ~17 min (o laço no
  host chama a política por mundo a cada passo; Dribbling a 4 800 passos domina), pico de
  RAM ~5 GB (um env CPU de set-up por mundo), GPU ~0.6 GB.
- O giro de 2 rad/s (caixa 2) segue no ticket 16 (needs-decision: o giro é caótico também na
  CPU).
