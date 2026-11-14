# 05: Critérios de aceitação do controlador de navegação

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Cenários novos no sweep e no `-m validation`, do repouso, alvos a 0.5 / 2 / 6 m e giros de 0 / 90 / 180°: chegada (erro final < 5 cm e < 2.5°), tempo (≤ 1.25× o trapézio analítico: 3.3 m/s, sobe a 1.30, freia a 1.0), chatter no alvo (RMS de velocidade no último segundo < 0.05 m/s) e teto (`ω` comandado ≤ 37.1). Os `spin` e `kick_recoil` são reescritos em target pose ou velocidade crua. Os números são ponto de partida: medir na CPU primeiro e fixar o limiar no medido mais margem.

**Blocked by:** 04 (Ação em target pose e controlador de navegação no backend CPU)

**Status:** done

- [x] Sweep roda a grade de alvos e giros e grava os CSVs
- [x] Limiares fixados a partir da medição CPU (registrar medido e margem neste ticket)
- [x] `test_validation` asserta chegada, tempo, chatter e teto; `spin` e `kick_recoil` ativos
- [x] `uv run pytest -m validation` verde

## Comments

**Feito em 2026-11-14** (branch `gpu-port-05-nav-acceptance`).

- **Cenário `navigate`** (`scripts/validation_sweep.py`): GoToBall, robô parado em (-3, 2)
  olhando +x, ação = target pose `d` m à frente girada `t`°, grade `d` ∈ {0.5, 2, 6} × `t` ∈
  {0, 90, 180}, 400 passos (10 s). Os 6 amarelos e a bola vão para a lateral de baixo, fora do
  caminho. Determinístico (estado inicial escrito inteiro), uma vez por parâmetro. Grava
  `navigation.csv` (e as colunas `nav_*` no `episodes.csv`).
- **Métricas**: erro final de posição e de heading; tempo de assentamento = primeiro instante a
  partir do qual o erro de posição fica < `NAV_POS_ERR` até o fim, dividido por
  `trapezoid_time(d)` (3.3 m/s, sobe a 1.30, freia a 1.0, triângulo quando não cruza; testado
  contra contas à mão); chatter = RMS de |v_xy| no último segundo (40 passos); teto = |ω| que o
  **controlador** devolve, gravado antes do clamp do `ctrlrange` (o `drive_cmd` recortado
  passaria por construção).
- **Medido na CPU (pior célula da grade) → limiar fixado:**

  | critério | ticket | medido | limiar | margem |
  |---|---|---|---|---|
  | erro final de posição | < 5 cm | 1.85 mm (0.5 m / 0°) | < 1 cm | 5.4× |
  | erro final de heading | < 2.5° | 2.37° (6 m / 180°) | < 2.5° | 0.13° |
  | tempo / trapézio | ≤ 1.25 | 0.950 (6 m) | ≤ 1.05 | 10 % |
  | chatter RMS | < 0.05 m/s | 0.0444 (0.5 m / 90°) | < 0.05 m/s | 12 % |
  | ω comandado | ≤ 37.1 | 6.283 (= 2π, giro de 180°) | ≤ 37.1 (`MAX_W`) | 5.9× |

- **Heading**: a zona morta de 2.5° é o próprio limite; o robô para logo dentro dela
  (2.27-2.37° nos giros de 90/180°). Não dá para apertar sem mudar o controlador.
- **Chatter**: o ±0.1 m/s sem banda morta dá 0.035-0.044 m/s RMS, abaixo de 0.05 sem mexer no
  controlador; margem fina (12 %). A grade é toda alinhada com +x; alvos em diagonal não foram
  medidos.
- **Tempo**: abaixo de 1 em todas as células porque a frenagem real (MAX_DV = 4 m/s²) é mais
  forte que o 1.0 planejado e "assentado" é a 1 cm, não parada total. O giro de 180° assenta em
  ~2.05 s (exponencial de ω = 2Δθ) e não entra no critério de tempo (o trapézio é de
  translação); `nav_yaw_settle_s` fica no CSV.
- **`spin`/`kick_recoil`**: já em velocidade crua (`step_raw`) desde o 04, seguem asserted em
  `test_scripted_scenario`.
- Sanidade: com `NAV_CHATTER_RMS = 0.03` as 9 células falham.
- `summarise` pula `impulse.csv` vazio (navigate sozinho não tem contato).
- **Verificação:** `uv run pytest` 94 passed; `-m validation` 29 passed; `validation_sweep
  --seeds 0-4` (todos os cenários, sob `MemoryMax=8G`) grava os 5 CSVs. `git merge
  gpu-port-impl`: already up to date.
