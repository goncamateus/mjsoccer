# Aplicar a correção dos roletes do `19` em `models/`

Type: task
Status: resolved
Blocked by: —

## Pergunta

O [`19`](19-wheel-axial-scrub.md) fechou o arrasto axial: três coisas freiam o giro do rolete, e
tirar as três leva o tier 1 a 3.29 / 4.02 m/s (x / y) com arrasto ~0. A variante é a `ellip_c3_d0`
de [`../candidate-19/variants.py`](../candidate-19/variants.py). Em `models/robot_rollers.xml`, nos
80 roletes:

1. Colisão: o casco `r_c_roller_00` (`maxhullvert 32`) vira elipsoide `size="0.0063 0.01377 0.0063"`
   (semi-eixo longo no eixo do rolete), com `contype="0" conaffinity="4"`.
2. `condim="4"` → `condim="3"` nas geoms de colisão dos roletes.
3. `damping="1e-6"` → `damping="0"` nas juntas `roll_wheel*`.

Fazer:
- Aplicar, e rodar a suíte e o harness do `14`. Registrar o que muda em contagem de contatos e em
  steps/s (o `07` tem o orçamento de colisão).
- Medir de novo o critério 5 do `14` e o `accel.py` dele, e conferir que a bola ainda bate nos roletes
  (`contype 7` da bola contra `conaffinity 4`).
- Passar ao `12` o envelope medido. O esperado pelo `19` é `v_max` 3.29 / 4.02 m/s medidos (cinemáticos 3.30 /
  4.04) e `a_max` sustentado entre as rampas 1.6–2.0 / 1.2–1.4 m/s² (limites analíticos 1.95 / 1.33).

## Answer

Resolvido em 2026-10-18. As três edições do `19` foram aplicadas nos 80 roletes de
`models/robot_rollers.xml`: elipsoide `size="0.0063 0.01377 0.0063"` com `contype 0` / `conaffinity 4`,
`condim 3` e `damping 0`. O asset `r_c_roller_00` saiu porque nenhuma geom o usa mais. Medições em
[`../candidate-20/`](../candidate-20/), antes (cópia de `models/` em `2e4c196`) e depois, via
`MJSOCCER_MODELS`.

**Envelope do tier 1, para o `12`** ([`drive.txt`](../candidate-20/drive.txt), `drive.py` do `19`
em `models/`, com paredes):

- `v_max`: **3.29 m/s em x** (comando cheio, 3.28 já aos 2 s) e **4.02 em y** (rampa de 1 m/s²).
  Cinemáticos: 3.30 / 4.04. Os números são os mesmos do `19`.
- `a_max` sustentado: em x, rampas de 1.6 e 2.0 dão 3.14 e 3.29 m/s a 2 s. Em y, a rampa de 1.2
  acompanha até 4.02 m/s, e a de 1.4 colapsa aos 2.7 s (yaw 19 rad/s). A rampa de 1.2 só cai aos
  4.3 s porque bate na parede: sem paredes, segura 4.02. Os limites analíticos do `19` (1.95 / 1.33)
  continuam valendo.
- **`ω_max`: o giro cheio chega a 36.2 rad/s**, acima do `MAX_W` 35.9 ([`spin.txt`](../candidate-20/spin.txt)).
  O 35.9 supõe giro em torno do centro das rodas. Com os roletes livres, o giro cheio rola sem
  escorregar, a r/R 0.380 contra 0.378. O centro instantâneo fica em (−2.6, +0.8) mm, junto do CoM
  em (−3.1, +0.2) mm, porque a 36 rad/s a força centrípeta do CoM fora do centro (m·Ω²·e ≈ 12 N)
  passa de μmg = 6.6 N. Em torno desse ponto, as rodas traseiras ficam mais perto do eixo e
  permitem yaw maior. A 0.25 ms dá 36.1 ([`dt.txt`](../candidate-20/dt.txt)), então não é efeito do
  passo. Antes da correção dava 30.1, segurado pelo escorregamento.

**`MAX_W` não foi mexido aqui.** Ele é o `ω_max` do tier 0 e normaliza a observação (`|ω|` lê até
1.01), e o `12` é quem decide isso. Os três testes que afirmam yaw < `MAX_W` com comando cheio
(`test_full_spin_stays_under_max_w`, `spin` k = 1, `dribble_turn` ω = 50) ficam `xfail(strict)`
apontando para o `12`. O `strict` faz eles falharem quando o `12` mover o número.

**Contatos e custo (`07`).** Contatos rolete↔piso por substep, ação aleatória, seeds 0-4
([`contacts.txt`](../candidate-20/contacts.txt)): com um robô quase não muda (GoToBall 6.38 → 6.28).
Com vários robôs parados cai bastante, porque o casco parado no piso gera vários contatos e o
elipsoide gera um: StaticDefenders 112.9 → 78.0, ContestedPossession 21.1 → 17.5, PassEndurance
24.0 → 18.2. Vazão, melhor de 3 × 2000 steps, intercalado antes/depois
([`throughput.txt`](../candidate-20/throughput.txt)): GoToBall 314 → 327 (+4 %), ContestedPossession
151 → 165 (+10 %), StaticDefenders 53.6 → 62.4 (+16 %). O `throughput.csv` do sweep, contra o do
`14`: tier 1 de +3 % a +16 %, tier 0 ±2 %. O critério 7 continua passando, agora com folga.

**A bola ainda bate nos roletes.** `attribution.py ball` do `07`, 18 casos
([`ball_before.txt`](../candidate-20/ball_before.txt), [`ball_after.txt`](../candidate-20/ball_after.txt)):
há contato bola|rolete em 11 casos (eram 10), com impulso somado de 0.102 N·s (era 0.110). A
velocidade de saída da bola muda no máximo ±0.03 m/s. O `contype 7` da bola contra o `conaffinity 4`
colide como esperado.

**Critério 5 do `14`.** O `accel.py` do `14` ([`accel.txt`](../candidate-20/accel.txt)) dá 2.35 / 1.85 m/s²
(eram 2.47 / 1.87), com `v` a 2 s de 3.29 / 0.45 m/s (eram 2.14 / 0.43). No eixo comandado, o teste
mede 2.24 / 1.63: passa dentro de 1.53–3.45, mas o lateral fica com só 6 % de folga. A janela de
0.1 s mede o transiente da partida. O sustentado é 1.95 / 1.33 (`19`), abaixo da faixa, então o
critério precisa ser redefinido pelo `12`.

**Harness do `14`** ([`sweep/`](../candidate-20/sweep/), seeds 0-4): 0 tombamentos em todos os
cenários. Inclinação com ação aleatória no tier 1, p95/p99 ≤ 2.40 / 3.60° (eram 2.62 / 3.86 com 50
seeds). Fora dos giros cheios, o maior yaw é 33.5 rad/s (parede, 6 m/s, 30°). GoToBall competente
acerta 1.0 no tier 1. O pior pico é 19.6° no ContestedPossession competente (choque com o
adversário; p99 3.2°, abaixo de 15°). Suíte padrão: 95 passam, 1 xfail. `-m validation`: 50 passam,
2 xfail.

**Testes novos.** `test_top_speed` (`validation`) exige 0.97 × cinemático: 3.29 em x e 4.02 em y
com rampa de 1. Falhava antes da correção. O `test_roller_floor_friction_is_measured` passou a
filtrar por `conaffinity`, porque os roletes agora têm `contype 0`.

**Docs:** `docs/index.md` atualizado (velocidade alcançada, giro de 36.2, sensibilidade ao `dt`).
